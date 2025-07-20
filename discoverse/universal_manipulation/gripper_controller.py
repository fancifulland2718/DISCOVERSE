"""
夹爪控制器 - 统一不同夹爪实现方式的接口

支持三种夹爪实现模式：
1. tendon控制 (如airbot_play)
2. equality约束 (如panda) 
3. 单关节控制 (如ur5e)

支持从传感器数据获取夹爪状态
"""

import numpy as np
import mujoco
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

class GripperController(ABC):
    """夹爪控制器基类"""
    
    def __init__(self, gripper_config: Dict[str, Any], mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        self.config = gripper_config
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.ctrl_index = gripper_config["ctrl_index"]
        self.ctrl_range = gripper_config["ctrl_range"]
        
        # 获取传感器索引（如果配置中有的话）
        self.sensor_index = None
        if "sensor_name" in gripper_config:
            try:
                self.sensor_index = mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_SENSOR, gripper_config["sensor_name"]
                )
            except Exception as e:
                print(f"Warning: Could not find gripper sensor '{gripper_config['sensor_name']}': {e}")
        
    @abstractmethod
    def set_position(self, position: float) -> bool:
        """设置夹爪位置"""
        pass
        
    @abstractmethod  
    def get_position(self) -> float:
        """获取当前夹爪位置"""
        pass
        
    def get_sensor_position(self) -> Optional[float]:
        """从传感器数据获取夹爪位置"""
        if self.sensor_index is not None:
            try:
                return self.mj_data.sensordata[self.sensor_index]
            except Exception as e:
                print(f"Warning: Failed to read gripper sensor data: {e}")
        return None
        
    def open(self) -> float:
        """打开夹爪，返回控制值"""
        return self.config["default_position"]
        
    def close(self) -> float:
        """关闭夹爪，返回控制值"""
        return self.config["close_position"]
        
    def is_grasping(self) -> bool:
        """检测是否正在抓取"""
        current_pos = self.get_position()
        if current_pos is not None:
            return current_pos <= self.config.get("grasp_threshold", 0.5)
        return False

class TwoFingerTendonGripper(GripperController):
    """双指tendon夹爪控制器 (如AirBot Play)"""
    
    def __init__(self, gripper_config: Dict[str, Any], mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        super().__init__(gripper_config, mj_model, mj_data)
        self.qpos_indices = gripper_config.get("qpos_indices", [])
        
    def set_position(self, position: float) -> bool:
        """通过tendon控制器设置夹爪位置"""
        try:
            # 将位置归一化到控制范围
            normalized_pos = np.clip(position, self.ctrl_range[0], self.ctrl_range[1])
            self.mj_data.ctrl[self.ctrl_index] = normalized_pos
            return True
        except Exception as e:
            print(f"Tendon gripper control failed: {e}")
            return False
            
    def get_position(self) -> float:
        """获取当前夹爪位置"""
        # 优先使用传感器数据
        sensor_pos = self.get_sensor_position()
        if sensor_pos is not None:
            return sensor_pos
            
        # 如果没有传感器，尝试从qpos推断
        try:
            if self.qpos_indices:
                qpos_values = self.mj_data.qpos[self.qpos_indices]
                # 对于airbot_play: endleft和endright通过tendon反向耦合
                grip_opening = np.abs(qpos_values[0])  # 使用左夹爪的绝对值
                # 映射到控制范围
                return np.interp(grip_opening, [0, 0.04], self.ctrl_range)
            else:
                # 返回当前控制器值作为备选
                return self.mj_data.ctrl[self.ctrl_index]
        except Exception as e:
            print(f"Failed to get tendon gripper position: {e}")
            return 0.0

class TwoFingerEqualityGripper(GripperController):
    """双指equality约束夹爪控制器 (如Panda)"""
    
    def __init__(self, gripper_config: Dict[str, Any], mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        super().__init__(gripper_config, mj_model, mj_data)
        self.qpos_indices = gripper_config.get("qpos_indices", [])
        
    def set_position(self, position: float) -> bool:
        """通过general控制器设置夹爪位置"""
        try:
            # 直接设置控制器值 (控制器直接对应夹爪开合)
            normalized_pos = np.clip(position, self.ctrl_range[0], self.ctrl_range[1])
            self.mj_data.ctrl[self.ctrl_index] = normalized_pos
            return True
        except Exception as e:
            print(f"Equality gripper control failed: {e}")
            return False
            
    def get_position(self) -> float:
        """获取当前夹爪位置"""
        # 优先使用传感器数据
        sensor_pos = self.get_sensor_position()
        if sensor_pos is not None:
            return sensor_pos
            
        # 如果没有传感器，尝试从qpos获取
        try:
            if self.qpos_indices:
                # 对于equality约束，两个关节位置相等
                # 取第一个关节的位置作为夹爪开合程度
                return self.mj_data.qpos[self.qpos_indices[0]]
            else:
                # 返回当前控制器值作为备选
                return self.mj_data.ctrl[self.ctrl_index]
        except Exception as e:
            print(f"Failed to get equality gripper position: {e}")
            return 0.0

class TwoFingerSingleGripper(GripperController):
    """双指单关节控制夹爪控制器 (如UR5e, ARX系列)"""
    
    def __init__(self, gripper_config: Dict[str, Any], mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        super().__init__(gripper_config, mj_model, mj_data)
        self.qpos_indices = gripper_config.get("qpos_indices", [])
        self.primary_joint_name = gripper_config.get("primary_joint", "")
        
    def set_position(self, position: float) -> bool:
        """通过position控制器设置夹爪位置"""
        try:
            # 直接设置控制器值 (控制主关节，从关节被动跟随)
            normalized_pos = np.clip(position, self.ctrl_range[0], self.ctrl_range[1])
            self.mj_data.ctrl[self.ctrl_index] = normalized_pos
            return True
        except Exception as e:
            print(f"Single gripper control failed: {e}")
            return False
            
    def get_position(self) -> float:
        """获取当前夹爪位置"""
        # 优先使用传感器数据
        sensor_pos = self.get_sensor_position()
        if sensor_pos is not None:
            return sensor_pos
            
        # 如果没有传感器，尝试从qpos获取
        try:
            if self.qpos_indices and len(self.qpos_indices) > 1:
                # 取主控制关节的位置 (通常是右夹爪)
                return self.mj_data.qpos[self.qpos_indices[1]]  # right finger (primary)
            elif self.qpos_indices:
                # 只有一个关节索引
                return self.mj_data.qpos[self.qpos_indices[0]]
            else:
                # 返回当前控制器值作为备选
                return self.mj_data.ctrl[self.ctrl_index]
        except Exception as e:
            print(f"Failed to get single gripper position: {e}")
            return 0.0

def create_gripper_controller(gripper_config: Dict[str, Any], mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> GripperController:
    """工厂函数：根据配置创建相应的夹爪控制器"""
    
    gripper_type = gripper_config["type"]
    
    if gripper_type == "two_finger_tendon":
        return TwoFingerTendonGripper(gripper_config, mj_model, mj_data)
    elif gripper_type == "two_finger_equality":
        return TwoFingerEqualityGripper(gripper_config, mj_model, mj_data)
    elif gripper_type == "two_finger_single":
        return TwoFingerSingleGripper(gripper_config, mj_model, mj_data)
    else:
        raise ValueError(f"Unsupported gripper type: {gripper_type}")
