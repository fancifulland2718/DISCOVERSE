"""
Universal Manipulation Framework

通用机械臂操作框架，支持多种机械臂执行相同任务。

主要组件:
- RobotConfigLoader: 机械臂配置加载器
- TaskConfigLoader: 任务配置加载器  
- MinkIKSolver: Mink逆运动学求解器
- UniversalTaskExecutor: 通用任务执行器
- PrimitiveRegistry: 动作原语注册器

使用示例:
    from discoverse.universal_manipulation import UniversalTaskBase
    
    task = UniversalTaskBase(
        robot_config="configs/robots/panda.yaml",
        task_config="configs/tasks/place_object.yaml",
        env_config=env_config
    )
    
    success = task.run_task(
        source_object="block_green",
        target_location="bowl_pink"
    )
"""

__version__ = "1.0.0"
__author__ = "DISCOVERSE Team"

# 导入主要模块 (避免导入时的GLFW依赖)
from .robot_config import RobotConfigLoader
from .task_config import TaskConfigLoader

# 延迟导入需要MuJoCo的模块
try:
    from .mink_solver import MinkIKSolver
except ImportError:
    # Mink未安装或MuJoCo不可用
    MinkIKSolver = None

# 导入其他核心模块
try:
    from .robot_interface import RobotInterface, PandaRobotInterface
    from .executor import UniversalTaskExecutor, TaskExecutionResult
    from .task_base import UniversalTaskBase
except ImportError as e:
    print(f"Warning: Could not import advanced modules: {e}")
    RobotInterface = None
    PandaRobotInterface = None
    UniversalTaskExecutor = None
    TaskExecutionResult = None
    UniversalTaskBase = None

# 导入原语模块
from .primitives import PrimitiveRegistry

__all__ = [
    "RobotConfigLoader",
    "TaskConfigLoader", 
    "MinkIKSolver",
    "RobotInterface",
    "PandaRobotInterface",
    "UniversalTaskExecutor",
    "TaskExecutionResult",
    "UniversalTaskBase",
    "PrimitiveRegistry",
] 