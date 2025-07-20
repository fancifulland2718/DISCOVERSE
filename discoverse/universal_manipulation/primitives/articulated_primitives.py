"""
铰接操作原语

实现机械臂的铰接物体操作功能，如开门、拉抽屉等。
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple

from .base_primitive import BasePrimitive, PrimitiveResult, PrimitiveStatus

class OpenArticulatedPrimitive(BasePrimitive):
    """打开铰接物体原语"""
    
    def __init__(self):
        super().__init__(
            name="open_articulated",
            description="打开铰接物体（抽屉、门等）"
        )
    
    def get_required_parameters(self) -> List[str]:
        return ["object_name", "handle_site"]
    
    def get_optional_parameters(self) -> List[str]:
        return ["open_direction", "distance", "speed"]
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        if "object_name" not in params:
            return False, "Missing required parameter: object_name"
        
        if "handle_site" not in params:
            return False, "Missing required parameter: handle_site"
        
        return True, ""
    
    def execute(self, robot_interface, params: Dict[str, Any], timeout: float = 10.0) -> PrimitiveResult:
        start_time = time.time()
        
        try:
            object_name = params["object_name"]
            handle_site = params["handle_site"]
            open_direction = params.get("open_direction", [1, 0, 0])
            distance = params.get("distance", 0.2)
            speed = params.get("speed", 0.1)
            
            # 执行开启操作
            success = robot_interface.open_articulated(
                object_name, handle_site, open_direction, distance, speed
            )
            
            execution_time = time.time() - start_time
            
            if success:
                return PrimitiveResult(
                    status=PrimitiveStatus.SUCCESS,
                    message=f"Successfully opened {object_name}",
                    execution_time=execution_time
                )
            else:
                return PrimitiveResult(
                    status=PrimitiveStatus.FAILURE,
                    message=f"Failed to open {object_name}",
                    execution_time=execution_time
                )
                
        except Exception as e:
            return PrimitiveResult(
                status=PrimitiveStatus.ERROR,
                message=f"Error in open_articulated: {str(e)}",
                execution_time=time.time() - start_time,
                error_details={"exception": str(e)}
            )

class CloseArticulatedPrimitive(BasePrimitive):
    """关闭铰接物体原语"""
    
    def __init__(self):
        super().__init__(
            name="close_articulated",
            description="关闭铰接物体（抽屉、门等）"
        )
    
    def get_required_parameters(self) -> List[str]:
        return ["object_name", "handle_site"]
    
    def get_optional_parameters(self) -> List[str]:
        return ["close_direction", "distance"]
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        if "object_name" not in params:
            return False, "Missing required parameter: object_name"
        
        if "handle_site" not in params:
            return False, "Missing required parameter: handle_site"
        
        return True, ""
    
    def execute(self, robot_interface, params: Dict[str, Any], timeout: float = 10.0) -> PrimitiveResult:
        start_time = time.time()
        
        try:
            object_name = params["object_name"]
            handle_site = params["handle_site"]
            close_direction = params.get("close_direction", [-1, 0, 0])
            distance = params.get("distance", 0.2)
            
            # 执行关闭操作
            success = robot_interface.close_articulated(
                object_name, handle_site, close_direction, distance
            )
            
            execution_time = time.time() - start_time
            
            if success:
                return PrimitiveResult(
                    status=PrimitiveStatus.SUCCESS,
                    message=f"Successfully closed {object_name}",
                    execution_time=execution_time
                )
            else:
                return PrimitiveResult(
                    status=PrimitiveStatus.FAILURE,
                    message=f"Failed to close {object_name}",
                    execution_time=execution_time
                )
                
        except Exception as e:
            return PrimitiveResult(
                status=PrimitiveStatus.ERROR,
                message=f"Error in close_articulated: {str(e)}",
                execution_time=time.time() - start_time,
                error_details={"exception": str(e)}
            ) 