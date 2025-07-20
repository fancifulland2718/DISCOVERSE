"""
动作原语基类

定义所有动作原语的基础接口和通用功能。
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

class PrimitiveStatus(Enum):
    """原语执行状态"""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"
    TIMEOUT = "timeout"
    ERROR = "error"

@dataclass
class PrimitiveResult:
    """原语执行结果"""
    status: PrimitiveStatus
    message: str = ""
    execution_time: float = 0.0
    error_details: Optional[Dict[str, Any]] = None
    intermediate_data: Optional[Dict[str, Any]] = None
    
    @property
    def success(self) -> bool:
        """是否执行成功"""
        return self.status == PrimitiveStatus.SUCCESS
    
    @property
    def failed(self) -> bool:
        """是否执行失败"""
        return self.status in [PrimitiveStatus.FAILURE, PrimitiveStatus.TIMEOUT, PrimitiveStatus.ERROR]

class BasePrimitive(ABC):
    """动作原语基类"""
    
    def __init__(self, name: str, description: str = ""):
        """
        初始化动作原语
        
        Args:
            name: 原语名称
            description: 原语描述
        """
        self.name = name
        self.description = description
        self.execution_count = 0
        self.success_count = 0
        self.total_execution_time = 0.0
        
    @abstractmethod
    def execute(self, 
                robot_interface,  # 机械臂接口对象
                params: Dict[str, Any],
                timeout: float = 10.0) -> PrimitiveResult:
        """
        执行动作原语
        
        Args:
            robot_interface: 机械臂接口对象
            params: 执行参数
            timeout: 超时时间（秒）
            
        Returns:
            执行结果
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        验证参数有效性
        
        Args:
            params: 待验证的参数
            
        Returns:
            Tuple[是否有效, 错误信息]
        """
        pass
    
    @abstractmethod
    def get_required_parameters(self) -> List[str]:
        """
        获取必需参数列表
        
        Returns:
            必需参数名称列表
        """
        pass
    
    def get_optional_parameters(self) -> List[str]:
        """
        获取可选参数列表
        
        Returns:
            可选参数名称列表
        """
        return []
    
    def estimate_execution_time(self, params: Dict[str, Any]) -> float:
        """
        估算执行时间
        
        Args:
            params: 执行参数
            
        Returns:
            预估执行时间（秒）
        """
        return 5.0  # 默认5秒
    
    def can_execute(self, robot_interface, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        检查是否可以执行
        
        Args:
            robot_interface: 机械臂接口
            params: 执行参数
            
        Returns:
            Tuple[是否可执行, 错误信息]
        """
        # 验证参数
        param_valid, param_error = self.validate_parameters(params)
        if not param_valid:
            return False, f"Parameter validation failed: {param_error}"
        
        # 检查机械臂状态
        if not robot_interface.is_ready():
            return False, "Robot is not ready"
        
        return True, ""
    
    def execute_with_validation(self,
                               robot_interface,
                               params: Dict[str, Any],
                               timeout: float = 10.0) -> PrimitiveResult:
        """
        带验证的执行方法
        
        Args:
            robot_interface: 机械臂接口
            params: 执行参数
            timeout: 超时时间
            
        Returns:
            执行结果
        """
        start_time = time.time()
        
        try:
            # 检查是否可以执行
            can_exec, error_msg = self.can_execute(robot_interface, params)
            if not can_exec:
                return PrimitiveResult(
                    status=PrimitiveStatus.FAILURE,
                    message=error_msg,
                    execution_time=time.time() - start_time
                )
            
            # 执行原语
            result = self.execute(robot_interface, params, timeout)
            
            # 更新统计信息
            self.execution_count += 1
            self.total_execution_time += result.execution_time
            if result.success:
                self.success_count += 1
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return PrimitiveResult(
                status=PrimitiveStatus.ERROR,
                message=f"Primitive execution failed: {str(e)}",
                execution_time=execution_time,
                error_details={"exception": str(e), "type": type(e).__name__}
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取执行统计信息
        
        Returns:
            统计信息字典
        """
        if self.execution_count == 0:
            return {
                "execution_count": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0
            }
        
        return {
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / self.execution_count,
            "average_execution_time": self.total_execution_time / self.execution_count,
            "total_execution_time": self.total_execution_time
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.execution_count = 0
        self.success_count = 0
        self.total_execution_time = 0.0
    
    def __str__(self) -> str:
        """字符串表示"""
        stats = self.get_statistics()
        return f"{self.name}(executions={stats['execution_count']}, success_rate={stats['success_rate']:.2f})"
    
    def __repr__(self) -> str:
        """对象表示"""
        return f"BasePrimitive(name='{self.name}', description='{self.description}')"


class CoordinateTransformMixin:
    """坐标变换混入类"""
    
    def transform_position(self, 
                          position: np.ndarray, 
                          from_frame: str, 
                          to_frame: str,
                          robot_interface) -> np.ndarray:
        """
        坐标变换
        
        Args:
            position: 位置 [x, y, z]
            from_frame: 源坐标系
            to_frame: 目标坐标系
            robot_interface: 机械臂接口
            
        Returns:
            变换后的位置
        """
        if from_frame == to_frame:
            return position.copy()
        
        # 实现具体的坐标变换逻辑
        if from_frame == "world" and to_frame == "robot_base":
            return robot_interface.world_to_base(position)
        elif from_frame == "robot_base" and to_frame == "world":
            return robot_interface.base_to_world(position)
        elif from_frame == "object" and to_frame == "world":
            # 需要知道物体的位置和姿态
            object_pose = robot_interface.get_object_pose(from_frame)
            if object_pose is None:
                raise ValueError(f"Cannot find object frame: {from_frame}")
            return object_pose[:3, :3] @ position + object_pose[:3, 3]
        else:
            raise ValueError(f"Unsupported coordinate transform: {from_frame} -> {to_frame}")
    
    def get_approach_vector(self, direction: str) -> np.ndarray:
        """
        获取接近方向向量
        
        Args:
            direction: 方向名称
            
        Returns:
            方向向量 [x, y, z]
        """
        direction_map = {
            "top_down": np.array([0, 0, -1]),
            "bottom_up": np.array([0, 0, 1]),
            "front": np.array([1, 0, 0]),
            "back": np.array([-1, 0, 0]),
            "left": np.array([0, 1, 0]),
            "right": np.array([0, -1, 0])
        }
        
        if direction in direction_map:
            return direction_map[direction]
        else:
            # 尝试解析为数值向量
            try:
                if isinstance(direction, (list, tuple, np.ndarray)):
                    vec = np.array(direction)
                    if len(vec) == 3:
                        return vec / np.linalg.norm(vec)  # 归一化
                raise ValueError(f"Invalid approach direction: {direction}")
            except:
                raise ValueError(f"Invalid approach direction: {direction}")


class SafetyMixin:
    """安全检查混入类"""
    
    def check_workspace_safety(self, 
                              target_position: np.ndarray,
                              robot_interface,
                              workspace_type: str = "reachable") -> Tuple[bool, str]:
        """
        检查工作空间安全性
        
        Args:
            target_position: 目标位置
            robot_interface: 机械臂接口
            workspace_type: 工作空间类型
            
        Returns:
            Tuple[是否安全, 错误信息]
        """
        # 由于删除了workspace配置，默认允许所有位置
        return True, ""
    
    def check_velocity_limits(self, 
                            velocity: np.ndarray,
                            robot_interface) -> Tuple[bool, str]:
        """
        检查速度限制
        
        Args:
            velocity: 关节速度
            robot_interface: 机械臂接口
            
        Returns:
            Tuple[是否安全, 错误信息]
        """
        # 由于删除了velocity_limits配置，默认允许所有速度
        return True, "" 