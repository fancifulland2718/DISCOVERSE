"""
动作原语模块

提供通用的机械臂操作原语，支持多种机械臂执行相同的操作。

原语分类:
- MovementPrimitives: 基础移动原语（move_to_pose, move_relative等）
- ManipulationPrimitives: 抓取操作原语（grasp_object, release_object等）
- ArticulatedPrimitives: 铰接操作原语（open_articulated, close_articulated等）

使用示例:
    from discoverse.universal_manipulation.primitives import PrimitiveRegistry
    
    registry = PrimitiveRegistry()
    move_primitive = registry.get_primitive("move_to_object")
    success = move_primitive.execute(params={"object_name": "block", "offset": [0, 0, 0.1]})
"""

from .base_primitive import BasePrimitive, PrimitiveResult, PrimitiveStatus
from .primitive_registry import PrimitiveRegistry

# 导入具体的原语类
try:
    from .movement_primitives import (
        MoveToObjectPrimitive,
        MoveRelativePrimitive,
        MoveToPosePrimitive
    )
    from .manipulation_primitives import (
        GraspObjectPrimitive,
        ReleaseObjectPrimitive,
        SetGripperPrimitive
    )
    from .articulated_primitives import (
        OpenArticulatedPrimitive,
        CloseArticulatedPrimitive
    )
    
    # 导入便利函数
    from .primitive_registry import get_global_registry, register_primitive, get_primitive
    
except ImportError as e:
    print(f"Warning: Could not import some primitives: {e}")

__all__ = [
    "BasePrimitive",
    "PrimitiveResult",
    "PrimitiveStatus",
    "PrimitiveRegistry",
    "get_global_registry",
    "register_primitive",
    "get_primitive",
    "MoveToObjectPrimitive",
    "MoveRelativePrimitive", 
    "MoveToPosePrimitive",
    "GraspObjectPrimitive",
    "ReleaseObjectPrimitive",
    "SetGripperPrimitive",
    "OpenArticulatedPrimitive",
    "CloseArticulatedPrimitive",
] 