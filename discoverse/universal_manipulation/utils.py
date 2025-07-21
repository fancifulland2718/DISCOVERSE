"""
Universal Manipulation Utils

通用机械臂操作框架的工具函数集合
整合了原本分散在不同模块中的通用工具函数
"""

import numpy as np
import mujoco
from typing import Optional, Tuple, Dict, Any


def calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """计算两点之间的欧几里得距离
    
    Args:
        pos1: 第一个位置
        pos2: 第二个位置
        
    Returns:
        距离值
    """
    return np.linalg.norm(pos1 - pos2)


class SimpleStateMachine:
    """简单状态机实现"""
    
    def __init__(self):
        self.state_idx = 0
        self.max_state_cnt = 0
        self.triggered = False
        
    def reset(self):
        """重置状态机"""
        self.state_idx = 0
        self.triggered = False
        
    def trigger(self):
        """触发状态机，返回是否为新状态"""
        if not self.triggered:
            self.triggered = True
            return True
        return False
        
    def update(self):
        """更新状态机"""
        pass
        
    def next(self):
        """切换到下一个状态"""
        self.state_idx += 1
        self.triggered = False


def validate_mujoco_object(model: mujoco.MjModel, data: mujoco.MjData, 
                          object_name: str, object_type: str = "body") -> bool:
    """验证MuJoCo对象是否存在
    
    Args:
        model: MuJoCo模型
        data: MuJoCo数据
        object_name: 对象名称
        object_type: 对象类型 (body, geom, site等)
        
    Returns:
        是否存在
    """
    try:
        if object_type == "body":
            _ = data.body(object_name)
        elif object_type == "geom":
            _ = data.geom(object_name)
        elif object_type == "site":
            _ = data.site(object_name)
        else:
            return False
        return True
    except:
        return False


def step_func(current: float, target: float, step_size: float) -> float:
    """平滑步进函数，用于控制平滑过渡
    
    Args:
        current: 当前值
        target: 目标值
        step_size: 步长大小
        
    Returns:
        更新后的值
    """
    diff = target - current
    if abs(diff) <= step_size:
        return target
    return current + np.sign(diff) * step_size


def get_body_tmat(data: mujoco.MjData, body_name: str) -> np.ndarray:
    """获取物体的变换矩阵
    
    Args:
        data: MuJoCo数据对象
        body_name: 物体名称
        
    Returns:
        4x4变换矩阵
        
    Raises:
        ValueError: 当物体不存在时
    """
    try:
        body = data.body(body_name)
        pos = body.xpos.copy()
        quat = body.xquat.copy()
        
        # 构建变换矩阵
        tmat = np.eye(4)
        
        # 设置旋转部分（四元数转旋转矩阵）
        rotation_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(rotation_matrix, quat)
        tmat[:3, :3] = rotation_matrix.reshape(3, 3)
        
        # 设置平移部分
        tmat[:3, 3] = pos
        
        return tmat
    except Exception as e:
        raise ValueError(f"Failed to get transform matrix for body '{body_name}': {e}")


def get_site_tmat(data: mujoco.MjData, site_name: str) -> np.ndarray:
    """获取站点的变换矩阵
    
    Args:
        data: MuJoCo数据对象
        site_name: 站点名称
        
    Returns:
        4x4变换矩阵
        
    Raises:
        ValueError: 当站点不存在时
    """
    try:
        site = data.site(site_name)
        pos = site.xpos.copy()
        mat = site.xmat.copy().reshape(3, 3)
        
        # 构建变换矩阵
        tmat = np.eye(4)
        tmat[:3, :3] = mat
        tmat[:3, 3] = pos
        
        return tmat
    except Exception as e:
        raise ValueError(f"Failed to get transform matrix for site '{site_name}': {e}")


def validate_mujoco_object(data: mujoco.MjData, object_name: str, object_type: str = "body") -> bool:
    """验证MuJoCo对象是否存在
    
    Args:
        data: MuJoCo数据对象
        object_name: 对象名称
        object_type: 对象类型 ("body", "site", "geom", "joint")
        
    Returns:
        对象是否存在
    """
    try:
        if object_type == "body":
            _ = data.body(object_name)
        elif object_type == "site":
            _ = data.site(object_name)
        elif object_type == "geom":
            _ = data.geom(object_name)
        elif object_type == "joint":
            _ = data.joint(object_name)
        else:
            return False
        return True
    except:
        return False


def calculate_distance_2d(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """计算两点间的2D距离（忽略Z轴）
    
    Args:
        pos1: 第一个点的3D坐标
        pos2: 第二个点的3D坐标
        
    Returns:
        2D距离
    """
    return np.linalg.norm(pos1[:2] - pos2[:2])


def calculate_distance_3d(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """计算两点间的3D距离
    
    Args:
        pos1: 第一个点的3D坐标
        pos2: 第二个点的3D坐标
        
    Returns:
        3D距离
    """
    return np.linalg.norm(pos1 - pos2)


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """归一化四元数
    
    Args:
        quat: 四元数 [w, x, y, z]
        
    Returns:
        归一化后的四元数
    """
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat / norm


def format_pose_info(pos: np.ndarray, quat: Optional[np.ndarray] = None) -> str:
    """格式化位姿信息用于打印
    
    Args:
        pos: 位置 [x, y, z]
        quat: 四元数 [w, x, y, z] (可选)
        
    Returns:
        格式化的字符串
    """
    info = f"位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
    if quat is not None:
        info += f", 四元数: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]"
    return info


class MotionProfiler:
    """运动轨迹分析器"""
    
    def __init__(self):
        self.positions = []
        self.velocities = []
        self.accelerations = []
        self.timestamps = []
        
    def add_sample(self, position: np.ndarray, timestamp: float):
        """添加位置样本"""
        self.positions.append(position.copy())
        self.timestamps.append(timestamp)
        
        # 计算速度和加速度
        if len(self.positions) > 1:
            dt = timestamp - self.timestamps[-2]
            if dt > 0:
                velocity = (position - self.positions[-2]) / dt
                self.velocities.append(velocity)
                
                if len(self.velocities) > 1:
                    acceleration = (velocity - self.velocities[-2]) / dt
                    self.accelerations.append(acceleration)
                    
    def get_max_velocity(self) -> float:
        """获取最大速度"""
        if not self.velocities:
            return 0.0
        return max(np.linalg.norm(v) for v in self.velocities)
        
    def get_max_acceleration(self) -> float:
        """获取最大加速度"""
        if not self.accelerations:
            return 0.0
        return max(np.linalg.norm(a) for a in self.accelerations)
        
    def reset(self):
        """重置分析器"""
        self.positions.clear()
        self.velocities.clear()
        self.accelerations.clear()
        self.timestamps.clear()


# 导出常用函数
__all__ = [
    'SimpleStateMachine',
    'step_func',
    'get_body_tmat',
    'get_site_tmat',
    'validate_mujoco_object',
    'calculate_distance_2d',
    'calculate_distance_3d',
    'normalize_quaternion',
    'format_pose_info',
    'MotionProfiler'
]
