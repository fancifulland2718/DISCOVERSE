"""
机械臂配置加载器

用于加载和解析机械臂配置文件，提供统一的配置接口。
"""

import os
import yaml
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

class RobotConfigLoader:
    """机械臂配置加载器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = None
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Robot config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # 验证配置文件
            self._validate_config()
            
            # 处理配置
            self._process_config()
            
            return self.config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse robot config file {config_path}: {e}")
    
    def _validate_config(self):
        """验证配置文件的必要字段"""
        required_fields = [
            'robot_name',
            'kinematics',
            'gripper'
        ]
        
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field in robot config: {field}")
        
        # 验证运动学配置
        kinematics = self.config['kinematics']
        kinematics_required = ['base_link', 'end_effector_site', 'arm_joints']
        for field in kinematics_required:
            if field not in kinematics:
                raise ValueError(f"Missing required kinematics field: {field}")
        
        # 验证夹爪配置
        gripper = self.config['gripper']
        gripper_required = ['type', 'ctrl_dim', 'ctrl_index']
        for field in gripper_required:
            if field not in gripper:
                raise ValueError(f"Missing required gripper field: {field}")
    
    def _process_config(self):
        """处理配置，转换数据类型等"""
        # 处理特殊配置
        if self.config['robot_name'] == 'airbot_play' and 'airbot_specific' in self.config:
            airbot_config = self.config['airbot_specific']
            if 'joint_bias' in airbot_config:
                airbot_config['joint_bias'] = np.array(airbot_config['joint_bias'])
            if 'arm_rotation_matrix' in airbot_config:
                airbot_config['arm_rotation_matrix'] = np.array(airbot_config['arm_rotation_matrix'])
    
    # ============== 属性访问方法 ==============
    
    @property
    def robot_name(self) -> str:
        """获取机械臂名称"""
        return self.config['robot_name']
    
    @property
    def base_link(self) -> str:
        """获取基座链接名称"""
        return self.config['kinematics']['base_link']
    
    @property
    def end_effector_site(self) -> str:
        """获取末端执行器site名称"""
        return self.config['kinematics']['end_effector_site']
    
    @property
    def dof(self) -> int:
        """获取总自由度数 (兼容性, 使用qpos_dim)"""
        return self.config.get('qpos_dim', self.config['kinematics'].get('dof', 0))
    
    @property
    def qpos_dim(self) -> int:
        """获取qpos维度"""
        return self.config['kinematics']['qpos_dim']
    
    @property  
    def ctrl_dim(self) -> int:
        """获取控制器维度"""
        return self.config['kinematics']['ctrl_dim']
    
    @property
    def arm_joints(self) -> List[str]:
        """获取机械臂关节名称列表"""
        return self.config['kinematics']['arm_joint_names']
    
    @property
    def arm_joints_count(self) -> int:
        """获取机械臂关节数"""
        return self.config['kinematics']['arm_joints']
    
    @property
    def gripper(self) -> Dict:
        """获取夹爪配置"""
        return self.config['gripper']
    
    @property
    def control(self) -> Dict:
        """获取控制配置"""
        return self.config.get('control', {})
    
    @property
    def arm_joint_names(self) -> List[str]:
        """获取机械臂关节名称列表"""
        return self.config['kinematics'].get('arm_joint_names', [])
    
    @property
    def gripper_joint_names(self) -> List[str]:
        """获取夹爪关节名称列表"""
        return self.config['gripper'].get('joint_names', [])
    
    @property
    def gripper_joint_indices(self) -> List[int]:
        """获取夹爪关节索引"""
        indices = self.config['gripper']['joint_indices']
        return indices if isinstance(indices, list) else [indices]
    
    @property
    def gripper_range(self) -> List[float]:
        """获取夹爪开合范围"""
        return self.config['gripper']['range']
    
    @property
    def joint_limits(self) -> Dict[str, np.ndarray]:
        """获取关节限制（从MuJoCo模型的actuator_ctrlrange获取）"""
        # 这个方法现在应该从MuJoCo模型中获取限制
        # 实际实现需要在有mj_model的情况下调用
        # 返回空字典，实际使用应该在robot_interface中调用get_joint_limits_from_mujoco
        return {}
    
    @property
    def default_poses(self) -> Dict[str, np.ndarray]:
        """获取默认位姿（从MJCF文件的keyframe中读取）"""
        # 从MJCF文件中读取默认位姿
        if 'mjcf_model' in self.config:
            mjcf_path = self.config['mjcf_model']['base_path']
            # 这里应该实现从MJCF文件读取keyframe的逻辑
            # 暂时返回空字典，实际实现需要解析MJCF文件
            return {}
        return {}
    
    @property
    def ik_solver_config(self) -> Dict[str, Any]:
        """获取IK求解器配置"""
        return self.config.get('ik_solver', {})
    
    @property
    def workspace(self) -> Dict[str, Any]:
        """获取工作空间配置（已删除，返回空字典）"""
        return {}
    
    @property
    def safety_config(self) -> Dict[str, Any]:
        """获取安全配置（已删除，返回空字典）"""
        return {}
    
    @property
    def sensors(self) -> Dict[str, Any]:
        """获取传感器配置"""
        return self.config.get('sensors', {})
    
    @property
    def control(self) -> Dict[str, Any]:
        """获取控制配置"""
        return self.config.get('control', {})
    
    # ============== 便利方法 ==============
    
    def get_home_pose(self) -> Optional[np.ndarray]:
        """获取home位姿（从MJCF文件读取）"""
        # 从MJCF文件中读取home位姿
        if 'mjcf_model' in self.config:
            mjcf_path = self.config['mjcf_model']['base_path']
            # 这里应该实现从MJCF文件读取keyframe的逻辑
            # 暂时返回None，实际实现需要解析MJCF文件
            return None
        return None
    
    def get_ready_pose(self) -> Optional[np.ndarray]:
        """获取ready位姿（从MJCF文件读取）"""
        # 从MJCF文件中读取ready位姿
        if 'mjcf_model' in self.config:
            mjcf_path = self.config['mjcf_model']['base_path']
            # 这里应该实现从MJCF文件读取keyframe的逻辑
            # 暂时返回None，实际实现需要解析MJCF文件
            return None
        return None
    
    def is_position_in_workspace(self, position: np.ndarray, workspace_type: str = 'reachable') -> bool:
        """
        检查位置是否在工作空间内
        
        Args:
            position: 位置坐标 [x, y, z]
            workspace_type: 工作空间类型 ('reachable' 或 'recommended')
            
        Returns:
            是否在工作空间内
        """
        # 由于删除了workspace配置，默认允许所有位置
        return True
    
    def get_joint_limit(self, joint_idx: int, limit_type: str = 'position') -> Optional[List[float]]:
        """
        获取指定关节的限制（从MuJoCo模型的actuator_ctrlrange获取）
        
        Args:
            joint_idx: 关节索引
            limit_type: 限制类型 ('position', 'velocity', 'effort')
            
        Returns:
            关节限制 [min, max] 或 max_value
        """
        # 这个方法现在应该从MuJoCo模型中获取限制
        # 实际实现需要在有mj_model的情况下调用
        return None
    
    def validate_joint_position(self, joint_positions: np.ndarray) -> bool:
        """
        验证关节位置是否在限制范围内（从MuJoCo模型的actuator_ctrlrange获取）
        
        Args:
            joint_positions: 关节位置数组
            
        Returns:
            是否在限制范围内
        """
        # 由于现在从MuJoCo模型获取限制，这里暂时返回True
        # 实际验证应该在robot_interface中进行
        return True
    
    def __str__(self) -> str:
        """字符串表示"""
        if not self.config:
            return "RobotConfigLoader(not loaded)"
        
        return f"RobotConfigLoader({self.robot_name}, {self.arm_joints}DOF + gripper)"
    
    def __repr__(self) -> str:
        """对象表示"""
        return self.__str__()


def load_robot_config(config_path: str) -> RobotConfigLoader:
    """
    便利函数：加载机械臂配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置加载器实例
    """
    return RobotConfigLoader(config_path) 