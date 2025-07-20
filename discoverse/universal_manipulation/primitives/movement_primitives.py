"""
基础移动原语

实现机械臂的基本移动功能，包括移动到位姿、相对移动、移动到物体等。
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.spatial.transform import Rotation

from .base_primitive import BasePrimitive, PrimitiveResult, PrimitiveStatus, CoordinateTransformMixin, SafetyMixin

class MoveToObjectPrimitive(BasePrimitive, CoordinateTransformMixin, SafetyMixin):
    """移动到物体位置原语"""
    
    def __init__(self):
        super().__init__(
            name="move_to_object",
            description="移动到指定物体的位置，支持偏移和接近方向"
        )
    
    def get_required_parameters(self) -> List[str]:
        return ["object_name"]
    
    def get_optional_parameters(self) -> List[str]:
        return ["offset", "approach_direction", "orientation", "coordinate_system"]
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        # 检查必需参数
        if "object_name" not in params:
            return False, "Missing required parameter: object_name"
        
        # 检查偏移
        if "offset" in params:
            offset = params["offset"]
            if not isinstance(offset, (list, tuple, np.ndarray)) or len(offset) != 3:
                return False, "offset must be a 3D vector [x, y, z]"
        
        # 检查接近方向
        if "approach_direction" in params:
            direction = params["approach_direction"]
            if not isinstance(direction, (str, list, tuple, np.ndarray)):
                return False, "approach_direction must be a string or 3D vector"
        
        return True, ""
    
    def execute(self, robot_interface, params: Dict[str, Any], timeout: float = 10.0) -> PrimitiveResult:
        start_time = time.time()
        
        try:
            # 获取参数
            object_name = params["object_name"]
            offset = np.array(params.get("offset", [0, 0, 0]))
            approach_direction = params.get("approach_direction", "top_down")
            coordinate_system = params.get("coordinate_system", "world")
            
            # 获取物体位置
            object_pose = robot_interface.get_object_pose(object_name)
            if object_pose is None:
                return PrimitiveResult(
                    status=PrimitiveStatus.FAILURE,
                    message=f"Object '{object_name}' not found",
                    execution_time=time.time() - start_time
                )
            
            # 计算目标位置
            if coordinate_system == "object":
                # 在物体坐标系中的偏移
                target_pos = object_pose[:3, :3] @ offset + object_pose[:3, 3]
            else:
                # 在世界坐标系中的偏移
                target_pos = object_pose[:3, 3] + offset
            
            # 处理接近方向
            if isinstance(approach_direction, str):
                approach_vector = self.get_approach_vector(approach_direction)
            else:
                approach_vector = np.array(approach_direction)
                approach_vector = approach_vector / np.linalg.norm(approach_vector)
            
            # 设置目标姿态（末端执行器朝向接近方向的反方向）
            target_ori = self._compute_target_orientation(approach_vector, object_pose[:3, :3], robot_interface)
            
            # 安全检查
            safe, safety_msg = self.check_workspace_safety(target_pos, robot_interface)
            if not safe:
                return PrimitiveResult(
                    status=PrimitiveStatus.FAILURE,
                    message=f"Safety check failed: {safety_msg}",
                    execution_time=time.time() - start_time
                )
            
            # 执行移动
            success = robot_interface.move_to_pose(target_pos, target_ori, timeout=timeout)
            
            execution_time = time.time() - start_time
            
            if success:
                return PrimitiveResult(
                    status=PrimitiveStatus.SUCCESS,
                    message=f"Successfully moved to object '{object_name}'",
                    execution_time=execution_time,
                    intermediate_data={
                        "target_position": target_pos.tolist(),
                        "object_position": object_pose[:3, 3].tolist(),
                        "offset": offset.tolist()
                    }
                )
            else:
                return PrimitiveResult(
                    status=PrimitiveStatus.FAILURE,
                    message=f"Failed to move to object '{object_name}'",
                    execution_time=execution_time
                )
                
        except Exception as e:
            return PrimitiveResult(
                status=PrimitiveStatus.ERROR,
                message=f"Error in move_to_object: {str(e)}",
                execution_time=time.time() - start_time,
                error_details={"exception": str(e)}
            )
    
    def _compute_target_orientation(self, approach_vector: np.ndarray, object_orientation: np.ndarray, robot_interface) -> np.ndarray:
        """
        计算目标姿态
        
        Args:
            approach_vector: 接近方向向量
            object_orientation: 物体姿态矩阵
            robot_interface: 机器人接口
            
        Returns:
            目标姿态矩阵
        """
        # 获取当前末端执行器姿态，保持不变以避免大幅姿态调整
        try:
            # 获取当前末端执行器的姿态矩阵
            end_effector_site = robot_interface.robot_config.end_effector_site
            site_id = robot_interface.mj_model.site(end_effector_site).id
            current_ori = robot_interface.mj_data.site_xmat[site_id].reshape(3, 3).copy()
            
            print(f"   🤖 当前末端姿态:")
            print(f"      X轴: {current_ori[:, 0]}")
            print(f"      Y轴: {current_ori[:, 1]}")
            print(f"      Z轴: {current_ori[:, 2]}")
            
            # 使用当前姿态作为目标姿态，避免大幅度旋转
            print(f"   ✅ 使用当前姿态作为目标姿态（避免大幅度旋转）")
            return current_ori
            
        except Exception as e:
            print(f"   ⚠️ 获取当前姿态失败，使用计算姿态: {e}")
            
            # 备选方案：简单实现：末端执行器z轴与接近方向对齐
            z_axis = -approach_vector  # 末端执行器z轴指向接近方向的反方向
            
            # 构造其他轴
            if abs(z_axis[2]) < 0.9:
                x_axis = np.cross([0, 0, 1], z_axis)
            else:
                x_axis = np.cross([1, 0, 0], z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            y_axis = np.cross(z_axis, x_axis)
            
            target_ori = np.column_stack([x_axis, y_axis, z_axis])
            
            # 调试信息：打印计算的目标姿态
            print(f"   🎯 目标姿态矩阵:")
            print(f"      X轴: {x_axis}")
            print(f"      Y轴: {y_axis}")  
            print(f"      Z轴: {z_axis}")
            print(f"      接近向量: {approach_vector}")
            
            return target_ori


class MoveRelativePrimitive(BasePrimitive, CoordinateTransformMixin, SafetyMixin):
    """相对移动原语"""
    
    def __init__(self):
        super().__init__(
            name="move_relative",
            description="相对当前位置进行移动"
        )
    
    def get_required_parameters(self) -> List[str]:
        return ["offset"]
    
    def get_optional_parameters(self) -> List[str]:
        return ["coordinate_system", "keep_orientation"]
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        if "offset" not in params:
            return False, "Missing required parameter: offset"
        
        offset = params["offset"]
        if not isinstance(offset, (list, tuple, np.ndarray)) or len(offset) != 3:
            return False, "offset must be a 3D vector [x, y, z]"
        
        return True, ""
    
    def execute(self, robot_interface, params: Dict[str, Any], timeout: float = 10.0) -> PrimitiveResult:
        start_time = time.time()
        
        try:
            # 获取参数
            offset = np.array(params["offset"])
            coordinate_system = params.get("coordinate_system", "world")
            keep_orientation = params.get("keep_orientation", True)
            
            # 获取当前位姿 - 直接从MuJoCo获取，避免传感器数据不准确
            end_effector_site = robot_interface.robot_config.end_effector_site
            site_id = robot_interface.mj_model.site(end_effector_site).id
            current_pos = robot_interface.mj_data.site_xpos[site_id].copy()
            current_ori = robot_interface.mj_data.site_xmat[site_id].reshape(3, 3).copy()
            
            # 计算目标位置
            if coordinate_system == "world":
                target_pos = current_pos + offset
            elif coordinate_system == "end_effector":
                # 在末端执行器坐标系中的偏移
                target_pos = current_pos + current_ori @ offset
            elif coordinate_system == "robot_base":
                # 转换到基座坐标系
                offset_world = robot_interface.base_to_world(offset)
                target_pos = current_pos + offset_world
            else:
                raise ValueError(f"Unsupported coordinate system: {coordinate_system}")
            
            # 设置目标姿态
            if keep_orientation:
                target_ori = current_ori
                print(f"   ✅ 保持当前姿态 (keep_orientation=True)")
            else:
                target_ori = np.eye(3)  # 默认姿态
                print(f"   🎯 使用默认姿态 (keep_orientation=False)")
                
            print(f"   🤖 当前末端姿态:")
            print(f"      X轴: {current_ori[:, 0]}")
            print(f"      Y轴: {current_ori[:, 1]}")
            print(f"      Z轴: {current_ori[:, 2]}")
            print(f"   🎯 目标末端姿态:")
            print(f"      X轴: {target_ori[:, 0]}")
            print(f"      Y轴: {target_ori[:, 1]}")
            print(f"      Z轴: {target_ori[:, 2]}")
            
            # 安全检查
            safe, safety_msg = self.check_workspace_safety(target_pos, robot_interface)
            if not safe:
                return PrimitiveResult(
                    status=PrimitiveStatus.FAILURE,
                    message=f"Safety check failed: {safety_msg}",
                    execution_time=time.time() - start_time
                )
            
            # 执行移动
            success = robot_interface.move_to_pose(target_pos, target_ori, timeout=timeout)
            
            execution_time = time.time() - start_time
            
            if success:
                return PrimitiveResult(
                    status=PrimitiveStatus.SUCCESS,
                    message=f"Successfully moved relative by {offset}",
                    execution_time=execution_time,
                    intermediate_data={
                        "offset": offset.tolist(),
                        "coordinate_system": coordinate_system,
                        "start_position": current_pos.tolist(),
                        "target_position": target_pos.tolist()
                    }
                )
            else:
                return PrimitiveResult(
                    status=PrimitiveStatus.FAILURE,
                    message=f"Failed to move relative by {offset}",
                    execution_time=execution_time
                )
                
        except Exception as e:
            return PrimitiveResult(
                status=PrimitiveStatus.ERROR,
                message=f"Error in move_relative: {str(e)}",
                execution_time=time.time() - start_time,
                error_details={"exception": str(e)}
            )


class MoveToPosePrimitive(BasePrimitive, SafetyMixin):
    """移动到绝对位姿原语"""
    
    def __init__(self):
        super().__init__(
            name="move_to_pose",
            description="移动到指定的绝对位姿"
        )
    
    def get_required_parameters(self) -> List[str]:
        return ["target_frame"]
    
    def get_optional_parameters(self) -> List[str]:
        return ["offset", "orientation", "coordinate_system"]
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        if "target_frame" not in params:
            return False, "Missing required parameter: target_frame"
        
        if "offset" in params:
            offset = params["offset"]
            if not isinstance(offset, (list, tuple, np.ndarray)) or len(offset) != 3:
                return False, "offset must be a 3D vector [x, y, z]"
        
        return True, ""
    
    def execute(self, robot_interface, params: Dict[str, Any], timeout: float = 10.0) -> PrimitiveResult:
        start_time = time.time()
        
        try:
            # 获取参数
            target_frame = params["target_frame"]
            offset = np.array(params.get("offset", [0, 0, 0]))
            coordinate_system = params.get("coordinate_system", "world")
            
            # 获取目标位姿
            if isinstance(target_frame, str):
                # 从frame名称获取位姿
                target_pose = robot_interface.get_frame_pose(target_frame)
                if target_pose is None:
                    return PrimitiveResult(
                        status=PrimitiveStatus.FAILURE,
                        message=f"Frame '{target_frame}' not found",
                        execution_time=time.time() - start_time
                    )
                target_pos = target_pose[:3, 3]
                target_ori = target_pose[:3, :3]
            else:
                # 直接使用位置坐标
                target_pos = np.array(target_frame)
                target_ori = params.get("orientation", np.eye(3))
            
            # 应用偏移
            target_pos = target_pos + offset
            
            # 安全检查
            safe, safety_msg = self.check_workspace_safety(target_pos, robot_interface)
            if not safe:
                return PrimitiveResult(
                    status=PrimitiveStatus.FAILURE,
                    message=f"Safety check failed: {safety_msg}",
                    execution_time=time.time() - start_time
                )
            
            # 执行移动
            success = robot_interface.move_to_pose(target_pos, target_ori, timeout=timeout)
            
            execution_time = time.time() - start_time
            
            if success:
                return PrimitiveResult(
                    status=PrimitiveStatus.SUCCESS,
                    message=f"Successfully moved to pose '{target_frame}'",
                    execution_time=execution_time,
                    intermediate_data={
                        "target_frame": str(target_frame),
                        "target_position": target_pos.tolist(),
                        "offset": offset.tolist()
                    }
                )
            else:
                return PrimitiveResult(
                    status=PrimitiveStatus.FAILURE,
                    message=f"Failed to move to pose '{target_frame}'",
                    execution_time=execution_time
                )
                
        except Exception as e:
            return PrimitiveResult(
                status=PrimitiveStatus.ERROR,
                message=f"Error in move_to_pose: {str(e)}",
                execution_time=time.time() - start_time,
                error_details={"exception": str(e)}
            ) 