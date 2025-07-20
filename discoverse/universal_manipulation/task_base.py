"""
通用任务基类

提供便利的任务执行接口，整合所有组件。
"""

import os
import mujoco
from typing import Dict, Any, Optional
from pathlib import Path

from .robot_config import RobotConfigLoader
from .task_config import TaskConfigLoader
from .robot_interface import RobotInterface, PandaRobotInterface
from .executor import UniversalTaskExecutor, TaskExecutionResult
from .primitives import PrimitiveRegistry

class UniversalTaskBase:
    """通用任务基类"""
    
    def __init__(self, 
                 robot_config_path: str,
                 task_config_path: str,
                 mj_model: mujoco.MjModel,
                 mj_data: mujoco.MjData,
                 robot_interface: Optional[RobotInterface] = None,
                 primitive_registry: Optional[PrimitiveRegistry] = None):
        """
        初始化通用任务
        
        Args:
            robot_config_path: 机械臂配置文件路径
            task_config_path: 任务配置文件路径
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
            robot_interface: 机械臂接口（可选，会自动创建）
            primitive_registry: 原语注册器（可选，使用全局注册器）
        """
        # 加载配置
        self.robot_config = RobotConfigLoader(robot_config_path)
        self.task_config = TaskConfigLoader(task_config_path)
        
        # 创建机械臂接口
        if robot_interface is None:
            robot_interface = self._create_robot_interface(mj_model, mj_data)
        self.robot_interface = robot_interface
        
        # 创建任务执行器
        self.executor = UniversalTaskExecutor(
            robot_interface=self.robot_interface,
            task_config=self.task_config,
            primitive_registry=primitive_registry
        )
        
        # 存储模型引用
        self.mj_model = mj_model
        self.mj_data = mj_data
    
    def _create_robot_interface(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        """
        根据机械臂类型创建对应的接口
        
        Args:
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
            
        Returns:
            机械臂接口实例
        """
        robot_name = self.robot_config.robot_name.lower()
        
        if robot_name == "panda":
            return PandaRobotInterface(mj_model, mj_data)
        elif robot_name == "airbot_play":
            from .robot_interface import AirbotRobotInterface
            return AirbotRobotInterface(mj_model, mj_data)
        elif robot_name in ["arx_x5", "arx_l5", "piper", "ur5e", "rm65", "xarm7"]:
            # 对于新支持的机械臂，使用通用接口
            from .robot_interface import GenericRobotInterface
            return GenericRobotInterface(self.robot_config, mj_model, mj_data)
        else:
            # 对于其他机械臂，暂时抛出错误
            raise NotImplementedError(f"Robot '{robot_name}' interface not implemented yet")
    
    def run_task(self, 
                 runtime_params: Optional[Dict[str, Any]] = None,
                 start_from_state: int = 0,
                 timeout: float = 300.0,
                 **kwargs) -> TaskExecutionResult:
        """
        运行任务
        
        Args:
            runtime_params: 运行时参数
            start_from_state: 从哪个状态开始
            timeout: 超时时间
            **kwargs: 额外的运行时参数
            
        Returns:
            任务执行结果
        """
        # 合并参数
        if runtime_params is None:
            runtime_params = {}
        runtime_params.update(kwargs)
        
        # 执行任务
        return self.executor.execute_task(
            runtime_params=runtime_params,
            start_from_state=start_from_state,
            timeout=timeout
        )
    
    def check_success(self) -> bool:
        """检查任务是否成功"""
        return self.executor.execution_result.success
    
    def get_status(self) -> Dict[str, Any]:
        """获取任务状态"""
        return self.executor.get_current_status()
    
    def get_robot_debug_info(self) -> Dict[str, Any]:
        """获取机械臂调试信息"""
        return self.robot_interface.get_debug_info()
    
    def pause(self):
        """暂停任务"""
        self.executor.pause_task()
    
    def resume(self):
        """恢复任务"""
        self.executor.resume_task()
    
    def stop(self):
        """停止任务"""
        self.executor.stop_task()
    
    @staticmethod
    def create_from_configs(robot_name: str, 
                           task_name: str,
                           mj_model,
                           mj_data,
                           configs_root: Optional[str] = None) -> 'UniversalTaskBase':
        """
        便利函数：从配置名称创建任务
        
        Args:
            robot_name: 机械臂名称
            task_name: 任务名称
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
            configs_root: 配置文件根目录
            
        Returns:
            任务实例
        """
        if configs_root is None:
            from discoverse import DISCOVERSE_ROOT_DIR
            configs_root = os.path.join(DISCOVERSE_ROOT_DIR, "discoverse", "configs")
        
        robot_config_path = os.path.join(configs_root, "robots", f"{robot_name}.yaml")
        task_config_path = os.path.join(configs_root, "tasks", f"{task_name}.yaml")
        
        return UniversalTaskBase(
            robot_config_path=robot_config_path,
            task_config_path=task_config_path,
            mj_model=mj_model,
            mj_data=mj_data
        )
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"UniversalTaskBase({self.robot_config.robot_name}, {self.task_config.task_name})" 