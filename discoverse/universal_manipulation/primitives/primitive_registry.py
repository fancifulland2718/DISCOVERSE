"""
动作原语注册器

管理所有可用的动作原语，提供注册、查询和执行功能。
"""

import yaml
import os
from typing import Dict, Any, Optional, List, Type
from pathlib import Path

from .base_primitive import BasePrimitive, PrimitiveResult
from discoverse import DISCOVERSE_ROOT_DIR

class PrimitiveRegistry:
    """动作原语注册器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化原语注册器
        
        Args:
            config_path: 原语配置文件路径
        """
        self._primitives: Dict[str, BasePrimitive] = {}
        self._primitive_configs: Dict[str, Dict[str, Any]] = {}
        
        # 默认配置文件路径
        if config_path is None:
            config_path = os.path.join(DISCOVERSE_ROOT_DIR, "discoverse", "configs", "primitives", "manipulation_primitives.yaml")
        
        self.config_path = config_path
        
        # 加载原语配置
        if os.path.exists(config_path):
            self.load_primitive_configs(config_path)
        
        # 注册内置原语
        self._register_builtin_primitives()
    
    def load_primitive_configs(self, config_path: str):
        """
        加载原语配置文件
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'primitives' in config:
                self._primitive_configs = config['primitives']
                
        except Exception as e:
            print(f"Warning: Failed to load primitive config from {config_path}: {e}")
    
    def register_primitive(self, primitive: BasePrimitive, override: bool = False):
        """
        注册动作原语
        
        Args:
            primitive: 原语实例
            override: 是否覆盖已存在的原语
            
        Raises:
            ValueError: 原语已存在且不允许覆盖
        """
        if primitive.name in self._primitives and not override:
            raise ValueError(f"Primitive '{primitive.name}' already registered. Use override=True to replace.")
        
        self._primitives[primitive.name] = primitive
        print(f"Registered primitive: {primitive.name}")
    
    def register_primitive_class(self, 
                                primitive_class: Type[BasePrimitive], 
                                name: str,
                                description: str = "",
                                override: bool = False):
        """
        注册原语类
        
        Args:
            primitive_class: 原语类
            name: 原语名称
            description: 原语描述
            override: 是否覆盖已存在的原语
        """
        primitive = primitive_class(name=name, description=description)
        self.register_primitive(primitive, override)
    
    def unregister_primitive(self, name: str) -> bool:
        """
        注销动作原语
        
        Args:
            name: 原语名称
            
        Returns:
            是否成功注销
        """
        if name in self._primitives:
            del self._primitives[name]
            print(f"Unregistered primitive: {name}")
            return True
        return False
    
    def get_primitive(self, name: str) -> Optional[BasePrimitive]:
        """
        获取动作原语
        
        Args:
            name: 原语名称
            
        Returns:
            原语实例，如果不存在则返回None
        """
        return self._primitives.get(name)
    
    def has_primitive(self, name: str) -> bool:
        """
        检查是否存在指定原语
        
        Args:
            name: 原语名称
            
        Returns:
            是否存在
        """
        return name in self._primitives
    
    def list_primitives(self) -> List[str]:
        """
        列出所有注册的原语名称
        
        Returns:
            原语名称列表
        """
        return list(self._primitives.keys())
    
    def get_primitive_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取原语信息
        
        Args:
            name: 原语名称
            
        Returns:
            原语信息字典
        """
        primitive = self.get_primitive(name)
        if primitive is None:
            return None
        
        config = self._primitive_configs.get(name, {})
        
        return {
            "name": primitive.name,
            "description": primitive.description,
            "required_parameters": primitive.get_required_parameters(),
            "optional_parameters": primitive.get_optional_parameters(),
            "config": config,
            "statistics": primitive.get_statistics()
        }
    
    def execute_primitive(self, 
                         name: str,
                         robot_interface,
                         params: Dict[str, Any],
                         timeout: float = 10.0) -> PrimitiveResult:
        """
        执行动作原语
        
        Args:
            name: 原语名称
            robot_interface: 机械臂接口
            params: 执行参数
            timeout: 超时时间
            
        Returns:
            执行结果
            
        Raises:
            ValueError: 原语不存在
        """
        primitive = self.get_primitive(name)
        if primitive is None:
            raise ValueError(f"Primitive '{name}' not found in registry")
        
        return primitive.execute_with_validation(robot_interface, params, timeout)
    
    def validate_primitive_params(self, name: str, params: Dict[str, Any]) -> tuple[bool, str]:
        """
        验证原语参数
        
        Args:
            name: 原语名称
            params: 参数字典
            
        Returns:
            Tuple[是否有效, 错误信息]
        """
        primitive = self.get_primitive(name)
        if primitive is None:
            return False, f"Primitive '{name}' not found"
        
        return primitive.validate_parameters(params)
    
    def get_primitive_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取原语配置
        
        Args:
            name: 原语名称
            
        Returns:
            配置字典
        """
        return self._primitive_configs.get(name)
    
    def get_primitives_by_category(self, category: str) -> List[str]:
        """
        根据类别获取原语列表
        
        Args:
            category: 类别名称 (movement/manipulation/articulated等)
            
        Returns:
            原语名称列表
        """
        category_primitives = []
        
        for name, config in self._primitive_configs.items():
            # 从配置或原语名称推断类别
            if category.lower() in name.lower():
                category_primitives.append(name)
        
        return category_primitives
    
    def _register_builtin_primitives(self):
        """注册内置原语"""
        # 这里会注册基础的原语实现
        # 实际的原语类会在具体的primitive模块中定义
        
        # 延迟导入以避免循环依赖
        try:
            from .movement_primitives import (
                MoveToObjectPrimitive,
                MoveRelativePrimitive,
                MoveToPosePrimitive
            )
            
            self.register_primitive(MoveToObjectPrimitive(), override=True)
            self.register_primitive(MoveRelativePrimitive(), override=True)
            self.register_primitive(MoveToPosePrimitive(), override=True)
            
        except ImportError as e:
            print(f"Warning: Could not import movement primitives: {e}")
        
        try:
            from .manipulation_primitives import (
                GraspObjectPrimitive,
                ReleaseObjectPrimitive,
                SetGripperPrimitive
            )
            
            self.register_primitive(GraspObjectPrimitive(), override=True)
            self.register_primitive(ReleaseObjectPrimitive(), override=True)
            self.register_primitive(SetGripperPrimitive(), override=True)
            
        except ImportError as e:
            print(f"Warning: Could not import manipulation primitives: {e}")
        
        try:
            from .articulated_primitives import (
                OpenArticulatedPrimitive,
                CloseArticulatedPrimitive
            )
            
            self.register_primitive(OpenArticulatedPrimitive(), override=True)
            self.register_primitive(CloseArticulatedPrimitive(), override=True)
            
        except ImportError as e:
            print(f"Warning: Could not import articulated primitives: {e}")
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """
        获取注册器统计信息
        
        Returns:
            统计信息字典
        """
        total_primitives = len(self._primitives)
        total_executions = sum(p.execution_count for p in self._primitives.values())
        total_successes = sum(p.success_count for p in self._primitives.values())
        
        return {
            "total_primitives": total_primitives,
            "total_executions": total_executions,
            "total_successes": total_successes,
            "overall_success_rate": total_successes / total_executions if total_executions > 0 else 0.0,
            "primitive_names": self.list_primitives()
        }
    
    def reset_all_statistics(self):
        """重置所有原语的统计信息"""
        for primitive in self._primitives.values():
            primitive.reset_statistics()
    
    def export_primitive_configs(self, output_path: str):
        """
        导出原语配置到文件
        
        Args:
            output_path: 输出文件路径
        """
        export_data = {
            "primitives": {},
            "registry_info": {
                "total_primitives": len(self._primitives),
                "primitive_names": self.list_primitives()
            }
        }
        
        for name, primitive in self._primitives.items():
            export_data["primitives"][name] = {
                "description": primitive.description,
                "required_parameters": primitive.get_required_parameters(),
                "optional_parameters": primitive.get_optional_parameters(),
                "statistics": primitive.get_statistics()
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(export_data, f, default_flow_style=False, allow_unicode=True)
    
    def __len__(self) -> int:
        """返回注册的原语数量"""
        return len(self._primitives)
    
    def __contains__(self, name: str) -> bool:
        """检查是否包含指定原语"""
        return name in self._primitives
    
    def __iter__(self):
        """迭代原语名称"""
        return iter(self._primitives.keys())
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"PrimitiveRegistry({len(self._primitives)} primitives)"
    
    def __repr__(self) -> str:
        """对象表示"""
        return f"PrimitiveRegistry(primitives={list(self._primitives.keys())})"


# 全局注册器实例
_global_registry = None

def get_global_registry() -> PrimitiveRegistry:
    """
    获取全局原语注册器实例
    
    Returns:
        全局注册器实例
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PrimitiveRegistry()
    return _global_registry

def register_primitive(primitive: BasePrimitive, override: bool = False):
    """
    向全局注册器注册原语
    
    Args:
        primitive: 原语实例
        override: 是否覆盖已存在的原语
    """
    get_global_registry().register_primitive(primitive, override)

def get_primitive(name: str) -> Optional[BasePrimitive]:
    """
    从全局注册器获取原语
    
    Args:
        name: 原语名称
        
    Returns:
        原语实例
    """
    return get_global_registry().get_primitive(name) 