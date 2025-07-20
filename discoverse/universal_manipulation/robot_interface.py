"""
通用机械臂接口

定义标准的机械臂操作接口，连接抽象原语和实际的机械臂控制。
"""

import time
import numpy as np
import mujoco
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from scipy.spatial.transform import Rotation

from .robot_config import RobotConfigLoader
from .mink_solver import MinkIKSolver
from .gripper_controller import create_gripper_controller, GripperController

class RobotInterface(ABC):
    """通用机械臂接口基类"""
    
    def __init__(self, robot_config: RobotConfigLoader, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        """
        初始化机械臂接口
        
        Args:
            robot_config: 机械臂配置
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
        """
        self.robot_config = robot_config
        self.mj_model = mj_model
        self.mj_data = mj_data
        
        # 可视化器引用（可选）
        self.viewer = None
        
        # 初始化IK求解器
        self.ik_solver = MinkIKSolver(robot_config, mj_model, mj_data)
        
        # 初始化夹爪控制器
        from .gripper_controller import create_gripper_controller
        self.gripper_controller = create_gripper_controller(
            robot_config.gripper, mj_model, mj_data
        )
        
        # 获取关节索引
        self._setup_joint_indices()
        
        # 控制状态 - 使用新的维度配置
        self.ctrl_dim = robot_config.ctrl_dim
        self.qpos_dim = robot_config.qpos_dim
        self.arm_joints = robot_config.arm_joints
        
        self.target_qpos = np.zeros(self.qpos_dim)
        self.is_moving = False
        self.motion_tolerance = 0.02
        self.velocity_tolerance = 0.1
        
    def set_viewer(self, viewer):
        """设置可视化器引用"""
        self.viewer = viewer
        
    def _setup_joint_indices(self):
        """设置关节索引"""
        # 设置传感器索引映射
        self._setup_sensor_indices()
        
        # 设置执行器索引映射
        self._setup_actuator_indices()
        
        # 保留关节索引映射（用于兼容性）
        self.arm_joint_indices = []
        self.gripper_joint_indices = []
        
        # 机械臂关节
        for joint_name in self.robot_config.arm_joint_names:
            try:
                joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                self.arm_joint_indices.append(joint_id)
            except Exception as e:
                print(f"Warning: Could not find joint {joint_name}: {e}")
        
        # 夹爪关节（兼容性保留）
        self.gripper_joint_indices = []
        if hasattr(self.robot_config, 'gripper') and 'qpos_joints' in self.robot_config.gripper:
            for joint_name in self.robot_config.gripper['qpos_joints']:
                try:
                    joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    self.gripper_joint_indices.append(joint_id)
                except Exception as e:
                    print(f"Warning: Could not find gripper joint {joint_name}: {e}")
    
    def _setup_sensor_indices(self):
        """设置传感器索引映射"""
        self.sensor_indices = {
            'joint_pos': [],
            'joint_vel': [],
            'joint_torque': [],
            'end_effector': {}
        }
        
        # 关节位置传感器
        if hasattr(self.robot_config, 'sensors') and 'joint_pos_sensors' in self.robot_config.sensors:
            for sensor_name in self.robot_config.sensors['joint_pos_sensors']:
                try:
                    sensor_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                    self.sensor_indices['joint_pos'].append(sensor_id)
                except Exception as e:
                    print(f"Warning: Could not find joint position sensor {sensor_name}: {e}")
        
        # 关节速度传感器
        if hasattr(self.robot_config, 'sensors') and 'joint_vel_sensors' in self.robot_config.sensors:
            for sensor_name in self.robot_config.sensors['joint_vel_sensors']:
                try:
                    sensor_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                    self.sensor_indices['joint_vel'].append(sensor_id)
                except Exception as e:
                    print(f"Warning: Could not find joint velocity sensor {sensor_name}: {e}")
        
        # 关节力矩传感器
        if hasattr(self.robot_config, 'sensors') and 'joint_torque_sensors' in self.robot_config.sensors:
            for sensor_name in self.robot_config.sensors['joint_torque_sensors']:
                try:
                    sensor_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                    self.sensor_indices['joint_torque'].append(sensor_id)
                except Exception as e:
                    print(f"Warning: Could not find joint torque sensor {sensor_name}: {e}")
        
        # 末端执行器传感器
        if hasattr(self.robot_config, 'sensors') and 'end_effector_sensors' in self.robot_config.sensors:
            end_effector_sensors = self.robot_config.sensors['end_effector_sensors']
            for sensor_type, sensor_name in end_effector_sensors.items():
                try:
                    sensor_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                    self.sensor_indices['end_effector'][sensor_type] = sensor_id
                except Exception as e:
                    print(f"Warning: Could not find end effector sensor {sensor_name}: {e}")
    
    def _setup_actuator_indices(self):
        """设置执行器索引映射"""
        self.actuator_indices = []
        
        if hasattr(self.robot_config, 'control') and 'actuators' in self.robot_config.control:
            for actuator_config in self.robot_config.control['actuators']:
                try:
                    # 执行器配置可能是字典或字符串
                    if isinstance(actuator_config, dict):
                        actuator_name = actuator_config['name']
                    else:
                        actuator_name = actuator_config
                        
                    actuator_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                    self.actuator_indices.append(actuator_id)
                except Exception as e:
                    print(f"Warning: Could not find actuator {actuator_config}: {e}")
    
    def _get_sensor_data(self, sensor_type: str, index: int = None) -> np.ndarray:
        """获取传感器数据"""
        if sensor_type not in self.sensor_indices:
            return np.array([])
        
        if index is not None:
            if index < len(self.sensor_indices[sensor_type]):
                sensor_id = self.sensor_indices[sensor_type][index]
                return np.array([self.mj_data.sensordata[sensor_id]])
            else:
                return np.array([])
        else:
            # 返回所有该类型的传感器数据
            data = []
            for sensor_id in self.sensor_indices[sensor_type]:
                data.append(self.mj_data.sensordata[sensor_id])
            return np.array(data)
    
    def _get_end_effector_sensor_data(self, sensor_type: str) -> np.ndarray:
        """获取末端执行器传感器数据"""
        if sensor_type not in self.sensor_indices['end_effector']:
            return np.array([])
        
        sensor_id = self.sensor_indices['end_effector'][sensor_type]
        return np.array([self.mj_data.sensordata[sensor_id]])
    
    # ============== 基础状态查询 ==============
    
    def is_ready(self) -> bool:
        """检查机械臂是否准备就绪"""
        return not self.is_moving
    
    def get_current_joint_positions(self) -> np.ndarray:
        """获取当前关节位置（通过传感器）"""
        # 优先使用传感器数据
        if self.sensor_indices['joint_pos']:
            return self._get_sensor_data('joint_pos')[:self.arm_joints]
        else:
            # 回退到直接访问qpos
            return self.mj_data.qpos[self.arm_joint_indices].copy()
    
    def get_current_joint_velocities(self) -> np.ndarray:
        """获取当前关节速度（通过传感器）"""
        # 优先使用传感器数据
        if self.sensor_indices['joint_vel']:
            return self._get_sensor_data('joint_vel')[:self.arm_joints]
        else:
            # 回退到直接访问qvel
            return self.mj_data.qvel[self.arm_joint_indices].copy()
    
    def get_current_joint_torques(self) -> np.ndarray:
        """获取当前关节力矩（通过传感器）"""
        if self.sensor_indices['joint_torque']:
            return self._get_sensor_data('joint_torque')[:self.arm_joints]
        else:
            return np.array([])
    
    def get_current_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前末端执行器位姿（通过传感器）
        
        Returns:
            Tuple[位置, 姿态矩阵]
        """
        # 优先使用传感器数据
        if 'position' in self.sensor_indices['end_effector'] and 'orientation' in self.sensor_indices['end_effector']:
            # 获取位置传感器数据
            pos_data = self._get_end_effector_sensor_data('position')
            if len(pos_data) >= 3:
                position = pos_data[:3]
            else:
                position = np.array([0, 0, 0])
            
            # 获取姿态传感器数据
            quat_data = self._get_end_effector_sensor_data('orientation')
            if len(quat_data) >= 4:
                # 转换为旋转矩阵
                from scipy.spatial.transform import Rotation
                quat = quat_data[:4]
                # MuJoCo的四元数格式是[w,x,y,z]，scipy期望的是[x,y,z,w]
                quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
                rotation = Rotation.from_quat(quat_scipy)
                orientation = rotation.as_matrix()
            else:
                orientation = np.eye(3)
        else:
            # 回退到直接访问site数据
            site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.robot_config.end_effector_site)
            position = self.mj_data.site_xpos[site_id].copy()
            orientation = self.mj_data.site_xmat[site_id].reshape(3, 3).copy()
        
        return position, orientation
    
    def get_gripper_position(self) -> float:
        """获取夹爪位置（归一化值）"""
        return self.gripper_controller.get_position()
    
    # ============== 物体和环境查询 ==============
    
    def get_object_pose(self, object_name: str) -> Optional[np.ndarray]:
        """
        获取物体位姿
        
        Args:
            object_name: 物体名称
            
        Returns:
            4x4变换矩阵，如果物体不存在则返回None
        """
        try:
            body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, object_name)
            position = self.mj_data.xpos[body_id].copy()
            orientation = self.mj_data.xmat[body_id].reshape(3, 3).copy()
            
            pose = np.eye(4)
            pose[:3, :3] = orientation
            pose[:3, 3] = position
            return pose
        except:
            return None
    
    def get_frame_pose(self, frame_name: str) -> Optional[np.ndarray]:
        """
        获取坐标系位姿
        
        Args:
            frame_name: 坐标系名称
            
        Returns:
            4x4变换矩阵
        """
        # 尝试作为site
        try:
            site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, frame_name)
            position = self.mj_data.site_xpos[site_id].copy()
            orientation = self.mj_data.site_xmat[site_id].reshape(3, 3).copy()
            
            pose = np.eye(4)
            pose[:3, :3] = orientation
            pose[:3, 3] = position
            return pose
        except:
            pass
        
        # 尝试作为body
        try:
            body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
            position = self.mj_data.xpos[body_id].copy()
            orientation = self.mj_data.xmat[body_id].reshape(3, 3).copy()
            
            pose = np.eye(4)
            pose[:3, :3] = orientation
            pose[:3, 3] = position
            return pose
        except:
            return None
    
    # ============== 坐标变换 ==============
    
    def world_to_base(self, position: np.ndarray) -> np.ndarray:
        """世界坐标系到机械臂基座坐标系"""
        base_pose = self.get_frame_pose(self.robot_config.base_link)
        if base_pose is None:
            return position  # 如果无法获取基座位姿，直接返回
        
        # 逆变换
        base_inv = np.linalg.inv(base_pose)
        pos_homo = np.append(position, 1.0)
        base_pos = base_inv @ pos_homo
        return base_pos[:3]
    
    def base_to_world(self, position: np.ndarray) -> np.ndarray:
        """机械臂基座坐标系到世界坐标系"""
        base_pose = self.get_frame_pose(self.robot_config.base_link)
        if base_pose is None:
            return position
        
        pos_homo = np.append(position, 1.0)
        world_pos = base_pose @ pos_homo
        return world_pos[:3]
    
    # ============== 安全和工作空间检查 ==============
    
    def is_position_in_workspace(self, position: np.ndarray, workspace_type: str = "reachable") -> bool:
        """检查位置是否在工作空间内"""
        return self.robot_config.is_position_in_workspace(position, workspace_type)
    
    def check_collision(self, target_position: np.ndarray) -> bool:
        """检查是否会发生碰撞"""
        # 改进的实现：区分IK失败和真正的碰撞
        try:
            current_pos, current_ori = self.get_current_pose()
            current_qpos = self.get_current_joint_positions()
            current_qpos_full = np.zeros(self.qpos_dim)
            current_qpos_full[:len(current_qpos)] = current_qpos
            
            # 尝试求解IK
            solution, converged, info = self.ik_solver.solve_ik(
                target_position, current_ori, current_qpos_full
            )
            
            if not converged:
                # IK未收敛不一定是碰撞，可能是求解器问题
                # 检查位置误差，如果误差很大才认为是真正无法到达
                position_error = info.get('final_position_error', float('inf'))
                if position_error > 0.1:  # 10cm误差阈值
                    return True  # 位置差太远，可能确实无法到达
                else:
                    return False  # 误差小，可能只是求解器收敛问题
            
            # IK收敛，检查关节限制（使用MuJoCo的actuator_ctrlrange）
            return not self.validate_joint_position_with_mujoco(solution[:self.arm_joints])
            
        except Exception as e:
            # 打印调试信息
            print(f"Warning: Collision check failed: {e}")
            return False  # 检查失败时保守地允许运动
    
    def get_velocity_limits(self) -> Optional[np.ndarray]:
        """获取速度限制（已删除，返回None）"""
        return None
    
    def get_joint_limits_from_mujoco(self) -> Dict[str, np.ndarray]:
        """
        从MuJoCo模型的actuator_ctrlrange获取关节限制
        
        Returns:
            包含position, velocity, effort限制的字典
        """
        limits = {
            'position': [],
            'velocity': [],
            'effort': []
        }
        
        # 获取机械臂关节的执行器限制
        for i in range(self.arm_joints):
            if i < len(self.actuator_indices):
                actuator_id = self.actuator_indices[i]
                if actuator_id < self.mj_model.nu:
                    # 从actuator_ctrlrange获取位置限制
                    ctrl_range = self.mj_model.actuator_ctrlrange[actuator_id]
                    if not np.isnan(ctrl_range[0]) and not np.isnan(ctrl_range[1]):
                        limits['position'].append([ctrl_range[0], ctrl_range[1]])
                    else:
                        # 如果没有设置限制，使用默认值
                        limits['position'].append([-np.pi, np.pi])
                    
                    # 从actuator_frcrange获取力矩限制
                    frc_range = self.mj_model.actuator_frcrange[actuator_id]
                    if not np.isnan(frc_range[0]) and not np.isnan(frc_range[1]):
                        limits['effort'].append(frc_range[1])  # 使用最大值
                    else:
                        limits['effort'].append(100.0)  # 默认值
                else:
                    limits['position'].append([-np.pi, np.pi])
                    limits['effort'].append(100.0)
            else:
                limits['position'].append([-np.pi, np.pi])
                limits['effort'].append(100.0)
        
        # 转换为numpy数组
        for key in limits:
            limits[key] = np.array(limits[key])
        
        return limits
    
    def validate_joint_position_with_mujoco(self, joint_positions: np.ndarray) -> bool:
        """
        使用MuJoCo的actuator_ctrlrange验证关节位置是否在限制范围内
        
        Args:
            joint_positions: 关节位置数组
            
        Returns:
            是否在限制范围内
        """
        if len(joint_positions) != self.robot_config.arm_joints:
            return False
        
        for i, pos in enumerate(joint_positions):
            if i < len(self.actuator_indices):
                actuator_id = self.actuator_indices[i]
                if actuator_id < self.mj_model.nu:
                    ctrl_range = self.mj_model.actuator_ctrlrange[actuator_id]
                    if not np.isnan(ctrl_range[0]) and not np.isnan(ctrl_range[1]):
                        if pos < ctrl_range[0] or pos > ctrl_range[1]:
                            return False
        
        return True
    
    # ============== 运动控制 ==============
    
    def move_to_pose(self, target_position: np.ndarray, target_orientation: np.ndarray, 
                     timeout: float = 10.0) -> bool:
        """
        移动到目标位姿
        
        Args:
            target_position: 目标位置
            target_orientation: 目标姿态矩阵
            timeout: 超时时间
            
        Returns:
            是否成功到达
        """
        try:
            # 获取完整的模型qpos（包含所有自由度）
            full_current_qpos = self.mj_data.qpos.copy()
            
            # 求解IK
            solution, converged, solve_info = self.ik_solver.solve_ik(
                target_position, target_orientation, full_current_qpos
            )
            
            if not converged:
                print(f"IK failed to converge: {solve_info['final_position_error']:.6f}")
                return False
            
            # 设置目标关节位置
            self.target_qpos[:self.robot_config.arm_joints] = solution
            
            # 执行运动
            return self._execute_joint_motion(timeout)
            
        except Exception as e:
            print(f"Move to pose failed: {e}")
            return False
    
    def move_joints(self, target_joint_positions: np.ndarray, timeout: float = 10.0) -> bool:
        """
        移动关节到目标位置（通过执行器）
        
        Args:
            target_joint_positions: 目标关节位置
            timeout: 超时时间
            
        Returns:
            是否成功到达
        """
        if len(target_joint_positions) != self.robot_config.arm_joints:
            print(f"Error: Expected {self.robot_config.arm_joints} joint positions, got {len(target_joint_positions)}")
            return False
        
        # 验证关节限制（使用MuJoCo的actuator_ctrlrange）
        if not self.validate_joint_position_with_mujoco(target_joint_positions):
            print("Error: Target joint positions exceed joint limits")
            return False
        
        # 通过执行器设置目标位置
        if self.actuator_indices:
            for i, actuator_id in enumerate(self.actuator_indices[:self.robot_config.arm_joints]):
                if i < len(target_joint_positions):
                    self.mj_data.ctrl[actuator_id] = target_joint_positions[i]
        else:
            # 回退到直接设置qpos
            self.mj_data.qpos[self.arm_joint_indices] = target_joint_positions
        
        # 设置目标位置（用于跟踪）
        self.target_qpos[:len(target_joint_positions)] = target_joint_positions
        return self._execute_joint_motion(timeout)
    
    def _execute_joint_motion(self, timeout: float) -> bool:
        """执行关节运动（通过执行器）"""
        start_time = time.time()
        self.is_moving = True
        
        try:
            # 调试信息：打印目标位置
            print(f"🎯 目标关节位置: {self.target_qpos[:self.robot_config.arm_joints]}")
            initial_qpos = self.get_current_joint_positions()
            print(f"🔄 初始关节位置: {initial_qpos}")
            
            step_count = 0
            while time.time() - start_time < timeout:
                # 通过执行器设置控制信号
                if self.actuator_indices:
                    for i, actuator_id in enumerate(self.actuator_indices[:self.robot_config.arm_joints]):
                        if i < len(self.target_qpos):
                            self.mj_data.ctrl[actuator_id] = self.target_qpos[i]
                else:
                    # 回退到直接设置控制信号
                    self.mj_data.ctrl[:self.robot_config.arm_joints] = self.target_qpos[:self.robot_config.arm_joints]
                
                # 步进仿真
                mujoco.mj_step(self.mj_model, self.mj_data)
                step_count += 1
                
                # 同步 viewer（如果设置了）
                if self.viewer is not None:
                    self.viewer.sync()
                
                # 检查是否到达目标
                current_qpos = self.get_current_joint_positions()
                current_qvel = self.get_current_joint_velocities()
                
                position_error = np.linalg.norm(current_qpos - self.target_qpos[:self.robot_config.arm_joints])
                velocity_magnitude = np.linalg.norm(current_qvel)
                
                # 每1000步打印一次调试信息
                if step_count % 1000 == 0:
                    print(f"   步数 {step_count}: pos_err={position_error:.6f}, vel_mag={velocity_magnitude:.6f}")
                    print(f"   当前位置: {current_qpos}")
                    print(f"   控制信号: {self.mj_data.ctrl[:self.robot_config.arm_joints]}")
                
                if position_error < self.motion_tolerance and velocity_magnitude < self.velocity_tolerance:
                    print(f"✅ 运动完成 (步数: {step_count}, 误差: {position_error:.6f})")
                    self.is_moving = False
                    return True
                
                # 短暂延时
                time.sleep(0.001)
            
            # 超时
            print(f"⏰ 运动超时 (步数: {step_count})")
            final_qpos = self.get_current_joint_positions()
            final_error = np.linalg.norm(final_qpos - self.target_qpos[:self.robot_config.arm_joints])
            print(f"   最终位置误差: {final_error:.6f}")
            print(f"   最终位置: {final_qpos}")
            self.is_moving = False
            return False
            
        except Exception as e:
            self.is_moving = False
            print(f"Joint motion execution failed: {e}")
            return False
    
    # ============== 夹爪控制 ==============
    
    def set_gripper(self, state: str, position: Optional[float] = None, force: float = 0.5) -> bool:
        """
        设置夹爪状态（通过夹爪控制器）
        
        Args:
            state: 'open', 'close', 'position'
            position: 具体位置（当state='position'时）
            force: 夹取力度（暂未使用）
            
        Returns:
            是否成功
        """
        try:
            if state == "open":
                return self.gripper_controller.open()
            elif state == "close":
                return self.gripper_controller.close()
            elif state == "position" and position is not None:
                return self.gripper_controller.set_position(position)
            else:
                return False
                
        except Exception as e:
            print(f"Gripper control failed: {e}")
            return False
    
    def grasp_object(self, object_name: str, grasp_type: str = "pinch", force: float = 0.5) -> bool:
        """抓取物体"""
        return self.set_gripper("close", force=force)
    
    def release_object(self, release_type: str = "gentle") -> bool:
        """释放物体"""
        return self.set_gripper("open")
    
    # ============== 高级操作 ==============
    
    @abstractmethod
    def open_articulated(self, object_name: str, handle_site: str, 
                        open_direction: List[float], distance: float, speed: float = 0.1) -> bool:
        """打开铰接物体（需要子类实现）"""
        pass
    
    @abstractmethod
    def close_articulated(self, object_name: str, handle_site: str,
                         close_direction: List[float], distance: float) -> bool:
        """关闭铰接物体（需要子类实现）"""
        pass
    
    # ============== 调试和状态 ==============
    
    def get_debug_info(self) -> Dict[str, Any]:
        """获取调试信息"""
        current_pos, current_ori = self.get_current_pose()
        current_qpos = self.get_current_joint_positions()
        
        return {
            "robot_name": self.robot_config.robot_name,
            "current_pose": {
                "position": current_pos.tolist(),
                "orientation": current_ori.tolist()
            },
            "current_joint_positions": current_qpos.tolist(),
            "target_joint_positions": self.target_qpos[:self.robot_config.arm_joints].tolist(),
            "gripper_position": self.get_gripper_position(),
            "is_moving": self.is_moving,
            "ik_solver_stats": self.ik_solver.get_statistics()
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"RobotInterface({self.robot_config.robot_name}, ready={self.is_ready()})"


class AirbotRobotInterface(RobotInterface):
    """AirBot Play机械臂接口实现"""
    
    def __init__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, 
                 config_path: Optional[str] = None):
        """
        初始化AirBot Play机械臂接口
        
        Args:
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
            config_path: 配置文件路径
        """
        if config_path is None:
            import os
            from discoverse import DISCOVERSE_ROOT_DIR
            config_path = os.path.join(DISCOVERSE_ROOT_DIR, "discoverse", "configs", "robots", "airbot_play.yaml")
        
        robot_config = RobotConfigLoader(config_path)
        super().__init__(robot_config, mj_model, mj_data)
    
    def open_articulated(self, object_name: str, handle_site: str,
                        open_direction: List[float], distance: float, speed: float = 0.1) -> bool:
        """实现AirBot Play的铰接物体开启"""
        # TODO: 实现具体的铰接操作逻辑
        print(f"Opening {object_name} with AirBot Play (not implemented)")
        return False
    
    def close_articulated(self, object_name: str, handle_site: str,
                         close_direction: List[float], distance: float) -> bool:
        """实现AirBot Play的铰接物体关闭"""
        # TODO: 实现具体的铰接操作逻辑
        print(f"Closing {object_name} with AirBot Play (not implemented)")
        return False


class PandaRobotInterface(RobotInterface):
    """Panda机械臂接口实现"""
    
    def __init__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, 
                 config_path: Optional[str] = None):
        """
        初始化Panda机械臂接口
        
        Args:
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
            config_path: 配置文件路径
        """
        if config_path is None:
            import os
            from discoverse import DISCOVERSE_ROOT_DIR
            config_path = os.path.join(DISCOVERSE_ROOT_DIR, "discoverse", "configs", "robots", "panda.yaml")
        
        robot_config = RobotConfigLoader(config_path)
        super().__init__(robot_config, mj_model, mj_data)
    
    def open_articulated(self, object_name: str, handle_site: str,
                        open_direction: List[float], distance: float, speed: float = 0.1) -> bool:
        """实现Panda的铰接物体开启"""
        # TODO: 实现具体的铰接操作逻辑
        print(f"Opening {object_name} with Panda (not implemented)")
        return False
    
    def close_articulated(self, object_name: str, handle_site: str,
                         close_direction: List[float], distance: float) -> bool:
        """实现Panda的铰接物体关闭"""
        # TODO: 实现具体的铰接操作逻辑
        print(f"Closing {object_name} with Panda (not implemented)")
        return False 


class GenericRobotInterface(RobotInterface):
    """通用机械臂接口 - 支持配置文件驱动的多种机械臂"""
    
    def __init__(self, robot_config: RobotConfigLoader, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        """
        初始化通用机械臂接口
        
        Args:
            robot_config: 机械臂配置
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
        """
        super().__init__(robot_config, mj_model, mj_data)
        print(f"🤖 {robot_config.robot_name.upper()} 通用接口初始化完成")
    
    def open_articulated(self, object_name: str, handle_site: str,
                        open_direction: List[float], distance: float, speed: float = 0.1) -> bool:
        """实现通用的铰接物体开启"""
        # TODO: 实现具体的铰接操作逻辑
        print(f"Opening {object_name} with {self.robot_config.robot_name} (not implemented)")
        return False
    
    def close_articulated(self, object_name: str, handle_site: str,
                         close_direction: List[float], distance: float) -> bool:
        """实现通用的铰接物体关闭"""
        # TODO: 实现具体的铰接操作逻辑
        print(f"Closing {object_name} with {self.robot_config.robot_name} (not implemented)")
        return False