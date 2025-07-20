"""
通用任务执行器

负责执行基于配置文件定义的任务，协调原语执行和状态管理。
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .robot_interface import RobotInterface
from .task_config import TaskConfigLoader
from .primitives import PrimitiveRegistry, PrimitiveResult, PrimitiveStatus

class TaskStatus(Enum):
    """任务执行状态"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"
    PAUSED = "paused"

@dataclass
class TaskExecutionResult:
    """任务执行结果"""
    status: TaskStatus
    message: str = ""
    execution_time: float = 0.0
    completed_states: int = 0
    total_states: int = 0
    error_details: Optional[Dict[str, Any]] = None
    state_results: List[PrimitiveResult] = None
    
    def __post_init__(self):
        if self.state_results is None:
            self.state_results = []
    
    @property
    def success(self) -> bool:
        """是否执行成功"""
        return self.status == TaskStatus.SUCCESS
    
    @property
    def failed(self) -> bool:
        """是否执行失败"""
        return self.status in [TaskStatus.FAILURE, TaskStatus.TIMEOUT, TaskStatus.ERROR]
    
    @property
    def progress(self) -> float:
        """执行进度 (0.0 - 1.0)"""
        if self.total_states == 0:
            return 0.0
        return self.completed_states / self.total_states

class UniversalTaskExecutor:
    """通用任务执行器"""
    
    def __init__(self, 
                 robot_interface: RobotInterface,
                 task_config: TaskConfigLoader,
                 primitive_registry: Optional[PrimitiveRegistry] = None):
        """
        初始化任务执行器
        
        Args:
            robot_interface: 机械臂接口
            task_config: 任务配置
            primitive_registry: 原语注册器
        """
        self.robot_interface = robot_interface
        self.task_config = task_config
        
        if primitive_registry is None:
            from .primitives import get_global_registry
            primitive_registry = get_global_registry()
        
        self.primitive_registry = primitive_registry
        
        # 执行状态
        self.current_state_index = 0
        self.execution_result = TaskExecutionResult(
            status=TaskStatus.NOT_STARTED,
            total_states=len(task_config.states)
        )
        self.start_time = 0.0
        self.is_paused = False
        
        # 执行选项
        self.max_retries = 3
        self.retry_delay = 1.0
        self.state_timeout = 30.0
        
        # 回调函数
        self.on_state_start = None
        self.on_state_complete = None
        self.on_task_complete = None
    
    def execute_task(self, 
                    runtime_params: Optional[Dict[str, Any]] = None,
                    start_from_state: int = 0,
                    timeout: float = 300.0) -> TaskExecutionResult:
        """
        执行完整任务
        
        Args:
            runtime_params: 运行时参数
            start_from_state: 从哪个状态开始执行
            timeout: 总超时时间
            
        Returns:
            任务执行结果
        """
        self.start_time = time.time()
        self.current_state_index = start_from_state
        
        try:
            # 设置运行时参数
            if runtime_params:
                self.task_config.set_runtime_parameters(**runtime_params)
            
            # 获取解析后的状态
            resolved_states = self.task_config.get_resolved_states()
            
            # 更新执行结果
            self.execution_result = TaskExecutionResult(
                status=TaskStatus.RUNNING,
                total_states=len(resolved_states),
                completed_states=start_from_state
            )
            
            print(f"🚀 开始执行任务: {self.task_config.task_name}")
            print(f"   总状态数: {len(resolved_states)}")
            print(f"   运行时参数: {runtime_params}")
            
            # 执行每个状态
            for i in range(start_from_state, len(resolved_states)):
                if time.time() - self.start_time > timeout:
                    self.execution_result.status = TaskStatus.TIMEOUT
                    self.execution_result.message = f"Task timeout after {timeout}s"
                    break
                
                if self.is_paused:
                    self.execution_result.status = TaskStatus.PAUSED
                    break
                
                self.current_state_index = i
                state_config = resolved_states[i]
                
                # 执行状态
                success = self._execute_state(state_config, i)
                
                if not success:
                    self.execution_result.status = TaskStatus.FAILURE
                    self.execution_result.message = f"State {i} ({state_config['name']}) failed"
                    break
                
                self.execution_result.completed_states = i + 1
            
            # 检查最终状态
            if self.execution_result.status == TaskStatus.RUNNING:
                # 检查任务成功条件
                if self._check_task_success():
                    self.execution_result.status = TaskStatus.SUCCESS
                    self.execution_result.message = "Task completed successfully"
                else:
                    self.execution_result.status = TaskStatus.FAILURE
                    self.execution_result.message = "Task completed but success condition not met"
            
            self.execution_result.execution_time = time.time() - self.start_time
            
            # 调用完成回调
            if self.on_task_complete:
                self.on_task_complete(self.execution_result)
            
            print(f"✅ 任务执行完成: {self.execution_result.status.value}")
            print(f"   执行时间: {self.execution_result.execution_time:.2f}s")
            print(f"   完成状态: {self.execution_result.completed_states}/{self.execution_result.total_states}")
            
            return self.execution_result
            
        except Exception as e:
            execution_time = time.time() - self.start_time
            self.execution_result = TaskExecutionResult(
                status=TaskStatus.ERROR,
                message=f"Task execution error: {str(e)}",
                execution_time=execution_time,
                completed_states=self.current_state_index,
                total_states=len(self.task_config.states),
                error_details={"exception": str(e), "type": type(e).__name__}
            )
            
            print(f"❌ 任务执行错误: {e}")
            return self.execution_result
    
    def _execute_state(self, state_config: Dict[str, Any], state_index: int) -> bool:
        """
        执行单个状态
        
        Args:
            state_config: 状态配置
            state_index: 状态索引
            
        Returns:
            是否执行成功
        """
        state_name = state_config["name"]
        primitive_name = state_config["primitive"]
        params = state_config.get("params", {})
        
        print(f"🔄 执行状态 {state_index}: {state_name} ({primitive_name})")
        
        # 调用状态开始回调
        if self.on_state_start:
            self.on_state_start(state_index, state_config)
        
        # 处理夹爪状态
        gripper_state = state_config.get("gripper_state")
        if gripper_state:
            self._set_gripper_state(gripper_state)
        
        # 执行原语
        success = False
        for attempt in range(self.max_retries):
            try:
                # 执行原语
                result = self.primitive_registry.execute_primitive(
                    primitive_name,
                    self.robot_interface,
                    params,
                    timeout=self.state_timeout
                )
                
                self.execution_result.state_results.append(result)
                
                if result.success:
                    success = True
                    print(f"   ✅ 状态完成: {result.message}")
                    break
                else:
                    print(f"   ❌ 状态失败 (尝试 {attempt + 1}/{self.max_retries}): {result.message}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                
            except Exception as e:
                print(f"   ❌ 状态执行异常 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        # 调用状态完成回调
        if self.on_state_complete:
            self.on_state_complete(state_index, state_config, success)
        
        return success
    
    def _set_gripper_state(self, gripper_state: str):
        """设置夹爪状态"""
        try:
            if gripper_state in ["open", "close"]:
                self.robot_interface.set_gripper(gripper_state)
                print(f"   🤏 夹爪设置为: {gripper_state}")
        except Exception as e:
            print(f"   ⚠️ 夹爪控制失败: {e}")
    
    def _check_task_success(self) -> bool:
        """检查任务成功条件"""
        success_condition = self.task_config.config.get("success_condition")
        if not success_condition:
            return True  # 没有成功条件就认为成功
        
        condition_type = success_condition.get("type")
        condition_params = success_condition.get("params", {})
        
        try:
            if condition_type == "object_in_target":
                # 检查物体是否在目标位置
                source_object = condition_params.get("source_object")
                target_location = condition_params.get("target_location")
                tolerance = condition_params.get("tolerance", 0.05)
                
                if source_object and target_location:
                    object_pose = self.robot_interface.get_object_pose(source_object)
                    target_pose = self.robot_interface.get_object_pose(target_location)
                    
                    if object_pose is not None and target_pose is not None:
                        distance = np.linalg.norm(object_pose[:3, 3] - target_pose[:3, 3])
                        return distance < tolerance
            
            # 其他成功条件类型可以在这里添加
            return True
            
        except Exception as e:
            print(f"⚠️ 成功条件检查失败: {e}")
            return True  # 检查失败时默认认为成功
    
    def pause_task(self):
        """暂停任务执行"""
        self.is_paused = True
        print("⏸️ 任务已暂停")
    
    def resume_task(self):
        """恢复任务执行"""
        self.is_paused = False
        print("▶️ 任务已恢复")
    
    def stop_task(self):
        """停止任务执行"""
        self.execution_result.status = TaskStatus.FAILURE
        self.execution_result.message = "Task stopped by user"
        print("⏹️ 任务已停止")
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前执行状态"""
        return {
            "task_name": self.task_config.task_name,
            "status": self.execution_result.status.value,
            "progress": self.execution_result.progress,
            "current_state": self.current_state_index,
            "total_states": self.execution_result.total_states,
            "execution_time": time.time() - self.start_time if self.start_time > 0 else 0.0,
            "is_paused": self.is_paused,
            "robot_ready": self.robot_interface.is_ready()
        }
    
    def get_state_history(self) -> List[Dict[str, Any]]:
        """获取状态执行历史"""
        history = []
        for i, result in enumerate(self.execution_result.state_results):
            if i < len(self.task_config.states):
                state_config = self.task_config.states[i]
                history.append({
                    "state_index": i,
                    "state_name": state_config["name"],
                    "primitive": state_config["primitive"],
                    "status": result.status.value,
                    "message": result.message,
                    "execution_time": result.execution_time
                })
        return history
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"UniversalTaskExecutor({self.task_config.task_name}, {self.execution_result.status.value})" 