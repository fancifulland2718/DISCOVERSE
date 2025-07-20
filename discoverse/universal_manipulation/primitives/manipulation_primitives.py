"""
抓取操作原语

实现机械臂的抓取和释放功能。
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple

from .base_primitive import BasePrimitive, PrimitiveResult, PrimitiveStatus

class GraspObjectPrimitive(BasePrimitive):
    """抓取物体原语"""
    
    def __init__(self):
        super().__init__(
            name="grasp_object",
            description="抓取指定物体"
        )
    
    def get_required_parameters(self) -> List[str]:
        return ["object_name"]
    
    def get_optional_parameters(self) -> List[str]:
        return ["grasp_type", "force", "timeout"]
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        if "object_name" not in params:
            return False, "Missing required parameter: object_name"
        
        if "force" in params:
            force = params["force"]
            if not isinstance(force, (int, float)) or force < 0 or force > 1:
                return False, "force must be a number between 0 and 1"
        
        return True, ""
    
    def execute(self, robot_interface, params: Dict[str, Any], timeout: float = 10.0) -> PrimitiveResult:
        start_time = time.time()
        
        try:
            object_name = params["object_name"]
            grasp_type = params.get("grasp_type", "pinch")
            force = params.get("force", 0.5)
            
            # 执行抓取
            success = robot_interface.grasp_object(object_name, grasp_type, force)
            
            execution_time = time.time() - start_time
            
            if success:
                return PrimitiveResult(
                    status=PrimitiveStatus.SUCCESS,
                    message=f"Successfully grasped object '{object_name}'",
                    execution_time=execution_time
                )
            else:
                return PrimitiveResult(
                    status=PrimitiveStatus.FAILURE,
                    message=f"Failed to grasp object '{object_name}'",
                    execution_time=execution_time
                )
                
        except Exception as e:
            return PrimitiveResult(
                status=PrimitiveStatus.ERROR,
                message=f"Error in grasp_object: {str(e)}",
                execution_time=time.time() - start_time,
                error_details={"exception": str(e)}
            )

class ReleaseObjectPrimitive(BasePrimitive):
    """释放物体原语"""
    
    def __init__(self):
        super().__init__(
            name="release_object",
            description="释放当前抓取的物体"
        )
    
    def get_required_parameters(self) -> List[str]:
        return []
    
    def get_optional_parameters(self) -> List[str]:
        return ["release_type", "delay"]
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        return True, ""
    
    def execute(self, robot_interface, params: Dict[str, Any], timeout: float = 10.0) -> PrimitiveResult:
        start_time = time.time()
        
        try:
            release_type = params.get("release_type", "gentle")
            delay = params.get("delay", 0.0)
            
            # 执行释放
            success = robot_interface.release_object(release_type)
            
            # 等待延时
            if delay > 0:
                time.sleep(delay)
            
            execution_time = time.time() - start_time
            
            if success:
                return PrimitiveResult(
                    status=PrimitiveStatus.SUCCESS,
                    message="Successfully released object",
                    execution_time=execution_time
                )
            else:
                return PrimitiveResult(
                    status=PrimitiveStatus.FAILURE,
                    message="Failed to release object",
                    execution_time=execution_time
                )
                
        except Exception as e:
            return PrimitiveResult(
                status=PrimitiveStatus.ERROR,
                message=f"Error in release_object: {str(e)}",
                execution_time=time.time() - start_time,
                error_details={"exception": str(e)}
            )

class SetGripperPrimitive(BasePrimitive):
    """设置夹爪状态原语"""
    
    def __init__(self):
        super().__init__(
            name="set_gripper",
            description="设置夹爪开合状态"
        )
    
    def get_required_parameters(self) -> List[str]:
        return ["state"]
    
    def get_optional_parameters(self) -> List[str]:
        return ["position", "force"]
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        if "state" not in params:
            return False, "Missing required parameter: state"
        
        state = params["state"]
        if state not in ["open", "close", "position"]:
            return False, "state must be 'open', 'close', or 'position'"
        
        if state == "position" and "position" not in params:
            return False, "position parameter required when state is 'position'"
        
        return True, ""
    
    def execute(self, robot_interface, params: Dict[str, Any], timeout: float = 10.0) -> PrimitiveResult:
        start_time = time.time()
        
        try:
            state = params["state"]
            position = params.get("position", None)
            force = params.get("force", 0.5)
            
            # 执行夹爪控制
            success = robot_interface.set_gripper(state, position, force)
            
            execution_time = time.time() - start_time
            
            if success:
                return PrimitiveResult(
                    status=PrimitiveStatus.SUCCESS,
                    message=f"Successfully set gripper to {state}",
                    execution_time=execution_time
                )
            else:
                return PrimitiveResult(
                    status=PrimitiveStatus.FAILURE,
                    message=f"Failed to set gripper to {state}",
                    execution_time=execution_time
                )
                
        except Exception as e:
            return PrimitiveResult(
                status=PrimitiveStatus.ERROR,
                message=f"Error in set_gripper: {str(e)}",
                execution_time=time.time() - start_time,
                error_details={"exception": str(e)}
            ) 