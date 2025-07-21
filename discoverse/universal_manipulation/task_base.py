"""
通用任务基类

提供便利的任务执行接口，整合所有组件。
"""

import os
import mujoco
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from .robot_config import RobotConfigLoader
from .task_config import TaskConfigLoader
from .robot_interface import RobotInterface, PandaRobotInterface
from .executor import UniversalTaskExecutor, TaskExecutionResult
from .primitives import PrimitiveRegistry

class UniversalTaskBase:
    """通用任务基类"""
    
    def __init__(self, 
                 robot_config_path: str,
                 task_config_path: str,
                 mj_model: mujoco.MjModel,
                 mj_data: mujoco.MjData,
                 robot_interface: Optional[RobotInterface] = None,
                 primitive_registry: Optional[PrimitiveRegistry] = None):
        """
        初始化通用任务
        
        Args:
            robot_config_path: 机械臂配置文件路径
            task_config_path: 任务配置文件路径
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
            robot_interface: 机械臂接口（可选，会自动创建）
            primitive_registry: 原语注册器（可选，使用全局注册器）
        """
        # 加载配置
        self.robot_config = RobotConfigLoader(robot_config_path)
        self.task_config = TaskConfigLoader(task_config_path)
        
        # 创建机械臂接口
        if robot_interface is None:
            robot_interface = self._create_robot_interface(mj_model, mj_data)
        self.robot_interface = robot_interface
        
        # 创建任务执行器
        self.executor = UniversalTaskExecutor(
            robot_interface=self.robot_interface,
            task_config=self.task_config,
            primitive_registry=primitive_registry
        )
        
        # 存储模型引用
        self.mj_model = mj_model
        self.mj_data = mj_data
    
    def _create_robot_interface(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        """
        根据机械臂类型创建对应的接口
        
        Args:
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
            
        Returns:
            机械臂接口实例
        """
        robot_name = self.robot_config.robot_name.lower()
        
        if robot_name == "panda":
            return PandaRobotInterface(mj_model, mj_data)
        elif robot_name == "airbot_play":
            from .robot_interface import AirbotRobotInterface
            return AirbotRobotInterface(mj_model, mj_data)
        elif robot_name in ["arx_x5", "arx_l5", "piper", "ur5e", "rm65", "xarm7", "iiwa14"]:
            # 对于新支持的机械臂，使用通用接口
            from .robot_interface import GenericRobotInterface
            return GenericRobotInterface(self.robot_config, mj_model, mj_data)
        else:
            # 对于其他机械臂，暂时抛出错误
            raise NotImplementedError(f"Robot '{robot_name}' interface not implemented yet")
    
    def run_task(self, 
                 runtime_params: Optional[Dict[str, Any]] = None,
                 start_from_state: int = 0,
                 timeout: float = 300.0,
                 **kwargs) -> TaskExecutionResult:
        """
        运行任务
        
        Args:
            runtime_params: 运行时参数
            start_from_state: 从哪个状态开始
            timeout: 超时时间
            **kwargs: 额外的运行时参数
            
        Returns:
            任务执行结果
        """
        # 合并参数
        if runtime_params is None:
            runtime_params = {}
        runtime_params.update(kwargs)
        
        # 执行任务
        return self.executor.execute_task(
            runtime_params=runtime_params,
            start_from_state=start_from_state,
            timeout=timeout
        )
    
    def check_success(self) -> bool:
        """检查任务是否成功"""
        print(f"   🔍 开始检查任务成功条件...")
        
        # 首先检查任务配置中是否有成功检查配置
        if hasattr(self.task_config, 'success_check'):
            print(f"   📋 使用配置化成功检查")
            return self._check_configured_success()
        
        print(f"   📋 使用默认成功检查")
        # 否则使用默认的执行结果检查
        return self.executor.execution_result.success if hasattr(self.executor, 'execution_result') else False
    
    def _check_configured_success(self) -> bool:
        """根据配置文件检查成功条件"""
        try:
            success_config = self.task_config.success_check
            method = success_config.get('method', 'simple')
            print(f"   📊 成功检查方法: {method}")
            
            if method == 'custom':
                # 保留原有的硬编码检查作为后备
                return self._check_custom_success()
            elif method == 'simple':
                return self._check_simple_conditions(success_config)
            elif method == 'combined':
                return self._check_combined_conditions(success_config)
            else:
                print(f"警告：未知的成功检查方法: {method}")
                return False
        except Exception as e:
            print(f"   ❌ 配置化成功检查异常: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_simple_conditions(self, success_config) -> bool:
        """检查简单成功条件（单一条件检查）"""
        conditions = success_config.get('conditions', [])
        
        print(f"   📋 检查 {len(conditions)} 个简单条件...")
        for i, condition in enumerate(conditions):
            description = condition.get('description', f'条件{i+1}')
            result = self._evaluate_condition(condition)
            status = "✅ 通过" if result else "❌ 失败"
            print(f"     {i+1}. {description}: {status}")
            if not result:
                return False
        return True
    
    def _check_combined_conditions(self, success_config) -> bool:
        """检查组合成功条件（多条件逻辑组合）"""
        conditions = success_config.get('conditions', [])
        operator = success_config.get('operator', 'and')
        
        print(f"   📋 检查 {len(conditions)} 个组合条件 (操作符: {operator})...")
        results = []
        for i, condition in enumerate(conditions):
            description = condition.get('description', f'条件{i+1}')
            result = self._evaluate_condition(condition)
            results.append(result)
            status = "✅ 通过" if result else "❌ 失败"
            print(f"     {i+1}. {description}: {status}")
        
        if operator == 'and':
            final_result = all(results)
        elif operator == 'or':
            final_result = any(results)
        else:
            print(f"警告：未知的逻辑操作符: {operator}")
            return False
            
        print(f"   🔍 组合结果 ({operator}): {'✅ 通过' if final_result else '❌ 失败'}")
        return final_result
    
    def _evaluate_condition(self, condition) -> bool:
        """评估单个成功条件
        
        Args:
            condition (dict): 条件配置
            
        Returns:
            bool: 条件满足返回True，否则返回False
        """
        condition_type = condition.get('type')
        
        try:
            if condition_type == 'distance':
                return self._check_distance_condition(condition)
            elif condition_type == 'distance_2d':
                return self._check_distance_2d_condition(condition)
            elif condition_type == 'position':
                return self._check_position_condition(condition)
            elif condition_type == 'orientation':
                return self._check_orientation_condition(condition)
            elif condition_type == 'height':
                return self._check_height_condition(condition)
            else:
                print(f"警告：未知的条件类型: {condition_type}")
                return False
        except Exception as e:
            print(f"条件检查失败 ({condition.get('description', '未知条件')}): {e}")
            return False
    
    def _check_distance_condition(self, condition) -> bool:
        """检查3D距离条件"""
        obj1 = condition.get('object1')
        obj2 = condition.get('object2')
        threshold = condition.get('threshold', 0.1)
        
        pos1 = self.mj_data.body(obj1).xpos
        pos2 = self.mj_data.body(obj2).xpos
        distance = np.linalg.norm(pos1 - pos2)
        
        return distance < threshold
    
    def _check_distance_2d_condition(self, condition) -> bool:
        """检查2D距离条件（忽略Z轴）"""
        obj1 = condition.get('object1')
        obj2 = condition.get('object2')
        threshold = condition.get('threshold', 0.1)
        
        pos1 = self.mj_data.body(obj1).xpos[:2]  # 只取x,y坐标
        pos2 = self.mj_data.body(obj2).xpos[:2]
        distance = np.linalg.norm(pos1 - pos2)
        
        # 调试信息
        description = condition.get('description', '2D距离检查')
        print(f"       🔍 {description}: 实际距离={distance:.4f}m, 阈值={threshold}m")
        
        return distance < threshold
    
    def _check_position_condition(self, condition) -> bool:
        """检查位置条件"""
        obj = condition.get('object')
        axis = condition.get('axis', 'z')
        threshold = condition.get('threshold')
        operator = condition.get('operator', '>')
        
        if threshold is None:
            return False
            
        pos = self.mj_data.body(obj).xpos
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map.get(axis, 2)
        value = pos[axis_idx]
        
        # 调试信息
        description = condition.get('description', f'{axis}轴位置检查')
        print(f"       🔍 {description}: 实际值={value:.4f}, {operator}{threshold}")
        
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        else:
            return False
    
    def _check_orientation_condition(self, condition) -> bool:
        """检查方向条件"""
        obj = condition.get('object')
        axis = condition.get('axis', 'z')
        direction = condition.get('direction', 'up')
        threshold = condition.get('threshold', 0.9)
        
        # 获取物体的旋转四元数
        quat = self.mj_data.body(obj).xquat
        
        # 将四元数转换为旋转矩阵
        rotation_matrix = np.zeros(9)  # 使用一维数组
        import mujoco
        mujoco.mju_quat2Mat(rotation_matrix, quat)
        rotation_matrix = rotation_matrix.reshape((3, 3))  # 重塑为3x3矩阵
        
        # 获取物体的局部轴在世界坐标系中的方向
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map.get(axis, 2)
        object_axis = rotation_matrix[:, axis_idx]
        
        # 定义目标方向
        direction_map = {
            'up': np.array([0, 0, 1]),
            'down': np.array([0, 0, -1]),
            'forward': np.array([1, 0, 0]),
            'backward': np.array([-1, 0, 0]),
            'left': np.array([0, 1, 0]),
            'right': np.array([0, -1, 0])
        }
        target_direction = direction_map.get(direction, np.array([0, 0, 1]))
        
        # 计算点积（余弦值）
        dot_product = np.dot(object_axis, target_direction)
        return dot_product > threshold
    
    def _check_height_condition(self, condition) -> bool:
        """检查高度条件（Z轴位置的简化版本）"""
        return self._check_position_condition({
            'object': condition.get('object'),
            'axis': 'z',
            'threshold': condition.get('threshold'),
            'operator': condition.get('operator', '>')
        })
    
    def _check_custom_success(self) -> bool:
        """自定义成功检查方法"""
        task_name = self.task_config.task_name
        
        if task_name == "cover_cup":
            return self._check_cover_cup_success()
        elif task_name == "place_block":
            return self._check_place_block_success()
        else:
            # 未知任务，返回False
            return False
    
    def _check_cover_cup_success(self) -> bool:
        """检查cover_cup任务成功条件"""
        try:
            from discoverse.utils import get_body_tmat
            
            tmat_lid = get_body_tmat(self.mj_data, "cup_lid")
            tmat_cup = get_body_tmat(self.mj_data, "coffeecup_white")
            tmat_plate = get_body_tmat(self.mj_data, "plate_white")
            
            # 检查杯子是否直立 (Z轴朝上)
            cup_upright = abs(tmat_cup[2, 2]) > 0.99
            
            # 检查杯子是否在盘子上 (XY平面距离<2cm)
            cup_on_plate = np.hypot(tmat_plate[0, 3] - tmat_cup[0, 3], 
                                   tmat_plate[1, 3] - tmat_cup[1, 3]) < 0.02
            
            # 检查杯盖是否盖在杯子上 (XY平面距离<2cm)
            lid_on_cup = np.hypot(tmat_lid[0, 3] - tmat_cup[0, 3], 
                                 tmat_lid[1, 3] - tmat_cup[1, 3]) < 0.02
            
            return cup_upright and cup_on_plate and lid_on_cup
            
        except Exception as e:
            print(f"Cover cup success check failed: {e}")
            return False
    
    def _check_place_block_success(self) -> bool:
        """检查place_block任务成功条件"""
        try:
            block_pos = self.mj_data.body('block_green').xpos
            bowl_pos = self.mj_data.body('bowl_pink').xpos
            distance = np.linalg.norm(block_pos[:2] - bowl_pos[:2])  # 只检查XY平面
            return distance < 0.03  # 3cm容差
        except:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取任务状态"""
        return self.executor.get_current_status()
    
    def get_robot_debug_info(self) -> Dict[str, Any]:
        """获取机械臂调试信息"""
        return self.robot_interface.get_debug_info()
    
    def pause(self):
        """暂停任务"""
        self.executor.pause_task()
    
    def resume(self):
        """恢复任务"""
        self.executor.resume_task()
    
    def stop(self):
        """停止任务"""
        self.executor.stop_task()
    
    @staticmethod
    def create_from_configs(robot_name: str, 
                           task_name: str,
                           mj_model,
                           mj_data,
                           configs_root: Optional[str] = None) -> 'UniversalTaskBase':
        """
        便利函数：从配置名称创建任务
        
        Args:
            robot_name: 机械臂名称
            task_name: 任务名称
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
            configs_root: 配置文件根目录
            
        Returns:
            任务实例
        """
        if configs_root is None:
            from discoverse import DISCOVERSE_ROOT_DIR
            configs_root = os.path.join(DISCOVERSE_ROOT_DIR, "discoverse", "configs")
        
        robot_config_path = os.path.join(configs_root, "robots", f"{robot_name}.yaml")
        task_config_path = os.path.join(configs_root, "tasks", f"{task_name}.yaml")
        
        return UniversalTaskBase(
            robot_config_path=robot_config_path,
            task_config_path=task_config_path,
            mj_model=mj_model,
            mj_data=mj_data
        )
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"UniversalTaskBase({self.robot_config.robot_name}, {self.task_config.task_name})" 