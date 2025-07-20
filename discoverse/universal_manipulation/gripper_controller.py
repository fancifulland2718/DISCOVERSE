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
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod

def get_joint_range(mj_model: mujoco.MjModel, joint_name: str) -> Tuple[float, float]:
    """从MuJoCo模型中获取关节的运动范围"""
    try:
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        joint_range = mj_model.jnt_range[joint_id]
        return float(joint_range[0]), float(joint_range[1])
    except Exception as e:
        print(f"Warning: Could not get range for joint '{joint_name}': {e}")
        return 0.0, 1.0  # 默认范围

class GripperController(ABC):
    """夹爪控制器基类"""
    
    def __init__(self, gripper_config: Dict[str, Any], mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        self.config = gripper_config
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.ctrl_index = gripper_config["ctrl_index"]
        
        # 从配置文件获取控制范围，但优先使用从MJCF文件中读取的实际关节范围
        self.config_ctrl_range = gripper_config["ctrl_range"]
        
        # 如果配置中指定了关节名称，尝试从MJCF获取真实范围
        # 使用配置中指定的控制范围
        self.ctrl_range = self.config_ctrl_range
        print(f"Gripper control range: {self.ctrl_range}")
        
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
                # 根据配置中的primary_joint获取主控制关节位置
                primary_joint = self.config.get("primary_joint", "")
                if primary_joint and "gripper_left_joint" in primary_joint:
                    # 对于xarm7，使用gripper_left_joint的位置（索引6：gripper_left_joint）
                    if len(qpos_values) > 6:
                        return qpos_values[6]  # gripper_left_joint的值直接对应tendon控制值
                    
                # 对于传统airbot_play: endleft和endright通过tendon反向耦合
                grip_opening = np.abs(qpos_values[0])  # 使用左夹爪的绝对值
                # 动态映射到控制范围，而不是硬编码
                max_opening = self.ctrl_range[1] if self.ctrl_range[1] > 0.01 else 0.04
                return np.interp(grip_opening, [0, max_opening], self.ctrl_range)
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

class MultiFingerEqualityGripper(GripperController):
    """多指equality约束夹爪控制器 (如RM65)"""
    
    def __init__(self, gripper_config: Dict[str, Any], mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        super().__init__(gripper_config, mj_model, mj_data)
        self.qpos_indices = gripper_config.get("qpos_indices", [])
        
    def set_position(self, position: float) -> bool:
        """通过equality约束设置多指夹爪位置"""
        try:
            # 直接设置控制器值，equality约束会自动同步其他关节
            normalized_pos = np.clip(position, self.ctrl_range[0], self.ctrl_range[1])
            self.mj_data.ctrl[self.ctrl_index] = normalized_pos
            return True
        except Exception as e:
            print(f"Multi-finger equality gripper control failed: {e}")
            return False
            
    def get_position(self) -> float:
        """获取当前多指夹爪位置"""
        # 优先使用传感器数据
        sensor_pos = self.get_sensor_position()
        if sensor_pos is not None:
            return sensor_pos
            
        # 如果没有传感器，使用主控制关节的qpos
        try:
            if self.qpos_indices:
                # 使用主控制关节的位置 (gripper_joint1)
                return self.mj_data.qpos[self.qpos_indices[0]]
            else:
                # 返回当前控制器值作为备选
                return self.mj_data.ctrl[self.ctrl_index]
        except Exception as e:
            print(f"Failed to get multi-finger equality gripper position: {e}")
            return 0.0

class MultiFingerSingleGripper(GripperController):
    """多指单控制器夹爪控制器 (如RM65)"""
    
    def __init__(self, gripper_config: Dict[str, Any], mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        super().__init__(gripper_config, mj_model, mj_data)
        self.qpos_indices = gripper_config.get("qpos_indices", [])
        
    def set_position(self, position: float) -> bool:
        """通过单一控制器设置多指夹爪位置"""
        try:
            # 直接设置控制器值，其他关节通过机械连接跟随
            normalized_pos = np.clip(position, self.ctrl_range[0], self.ctrl_range[1])
            self.mj_data.ctrl[self.ctrl_index] = normalized_pos
            return True
        except Exception as e:
            print(f"Multi-finger single gripper control failed: {e}")
            return False
            
    def get_position(self) -> float:
        """获取当前多指夹爪位置"""
        # 优先使用传感器数据
        sensor_pos = self.get_sensor_position()
        if sensor_pos is not None:
            return sensor_pos
            
        # 如果没有传感器，尝试从主控制关节的qpos获取
        try:
            if self.qpos_indices:
                # 使用主控制关节的位置 (gripper_joint1)
                return self.mj_data.qpos[self.qpos_indices[0]]
            else:
                # 返回当前控制器值作为备选
                return self.mj_data.ctrl[self.ctrl_index]
        except Exception as e:
            print(f"Failed to get multi-finger gripper position: {e}")
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
    elif gripper_type == "multi_finger_single":
        return MultiFingerSingleGripper(gripper_config, mj_model, mj_data)
    elif gripper_type == "multi_finger_equality":
        return MultiFingerEqualityGripper(gripper_config, mj_model, mj_data)
    else:
        raise ValueError(f"Unsupported gripper type: {gripper_type}")
