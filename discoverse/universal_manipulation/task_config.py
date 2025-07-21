"""
任务配置加载器

用于加载和解析任务配置文件，提供统一的任务定义接口。
"""

import os
import yaml
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

class TaskConfigLoader:
    """任务配置加载器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化任务配置加载器
        
        Args:
            config_path: 任务配置文件路径
        """
        self.config_path = config_path
        self.config = None
        self.runtime_params = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载任务配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            任务配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Task config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # 验证配置文件
            self._validate_config()
            
            return self.config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse task config file {config_path}: {e}")
    
    def _validate_config(self):
        """验证任务配置文件的必要字段"""
        required_fields = [
            'task_name',
            'description',
            'states'
        ]
        
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field in task config: {field}")
        
        # 验证状态配置
        if not isinstance(self.config['states'], list) or len(self.config['states']) == 0:
            raise ValueError("Task must have at least one state")
        
        for i, state in enumerate(self.config['states']):
            if 'name' not in state:
                raise ValueError(f"State {i} missing required field: name")
            if 'primitive' not in state:
                raise ValueError(f"State {i} ({state['name']}) missing required field: primitive")
    
    def set_runtime_parameters(self, **params):
        """
        设置运行时参数
        
        Args:
            **params: 运行时参数键值对
        """
        self.runtime_params.update(params)
    
    def resolve_parameters(self, value: Any) -> Any:
        """
        解析参数，支持运行时参数替换
        
        Args:
            value: 待解析的值
            
        Returns:
            解析后的值
        """
        if isinstance(value, str):
            # 处理参数占位符 {param_name}
            pattern = r'\{([^}]+)\}'
            matches = re.findall(pattern, value)
            
            for match in matches:
                if match in self.runtime_params:
                    value = value.replace(f'{{{match}}}', str(self.runtime_params[match]))
                elif match in self.config.get('parameters', {}):
                    # 使用配置文件中的默认参数
                    default_value = self.config['parameters'][match]
                    value = value.replace(f'{{{match}}}', str(default_value))
            
        elif isinstance(value, dict):
            return {k: self.resolve_parameters(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.resolve_parameters(item) for item in value]
        
        return value
    
    def get_resolved_states(self) -> List[Dict[str, Any]]:
        """
        获取解析后的状态序列
        
        Returns:
            解析后的状态列表
        """
        resolved_states = []
        
        for state in self.config['states']:
            resolved_state = self.resolve_parameters(state.copy())
            resolved_states.append(resolved_state)
        
        return resolved_states
    
    # ============== 属性访问方法 ==============
    
    @property
    def task_name(self) -> str:
        """获取任务名称"""
        return self.config['task_name']
    
    @property
    def description(self) -> str:
        """获取任务描述"""
        return self.config['description']
    
    @property
    def version(self) -> str:
        """获取任务版本"""
        return self.config.get('version', '1.0')
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """获取任务参数"""
        return self.config.get('parameters', {})
    
    @property
    def states(self) -> List[Dict[str, Any]]:
        """获取原始状态列表"""
        return self.config['states']
    
    @property
    def success_condition(self) -> Optional[Dict[str, Any]]:
        """获取成功条件"""
        return self.config.get('success_condition')
    
    @property 
    def success_check(self) -> Optional[Dict[str, Any]]:
        """获取成功检查配置"""
        return self.config.get('success_check')
    
    @property
    def failure_conditions(self) -> List[Dict[str, Any]]:
        """获取失败条件列表"""
        return self.config.get('failure_conditions', [])
    
    @property
    def safety_config(self) -> Dict[str, Any]:
        """获取安全配置"""
        return self.config.get('safety', {})
    
    @property
    def variants(self) -> Dict[str, Any]:
        """获取任务变体"""
        return self.config.get('variants', {})
    
    # ============== 便利方法 ==============
    
    def get_state_by_name(self, state_name: str) -> Optional[Dict[str, Any]]:
        """
        根据名称获取状态
        
        Args:
            state_name: 状态名称
            
        Returns:
            状态配置字典
        """
        for state in self.states:
            if state['name'] == state_name:
                return self.resolve_parameters(state.copy())
        return None
    
    def get_states_by_primitive(self, primitive_name: str) -> List[Dict[str, Any]]:
        """
        根据原语名称获取所有相关状态
        
        Args:
            primitive_name: 原语名称
            
        Returns:
            状态列表
        """
        matching_states = []
        for state in self.states:
            if state.get('primitive') == primitive_name:
                matching_states.append(self.resolve_parameters(state.copy()))
        return matching_states
    
    def get_required_parameters(self) -> List[str]:
        """
        获取必需的运行时参数列表
        
        Returns:
            参数名称列表
        """
        required_params = set()
        
        def extract_params(value):
            if isinstance(value, str):
                pattern = r'\{([^}]+)\}'
                matches = re.findall(pattern, value)
                for match in matches:
                    # 检查是否在默认参数中
                    if match not in self.config.get('parameters', {}):
                        required_params.add(match)
            elif isinstance(value, dict):
                for v in value.values():
                    extract_params(v)
            elif isinstance(value, list):
                for item in value:
                    extract_params(item)
        
        # 遍历所有状态
        for state in self.states:
            extract_params(state)
        
        # 遍历成功和失败条件
        if self.success_condition:
            extract_params(self.success_condition)
        
        for failure_condition in self.failure_conditions:
            extract_params(failure_condition)
        
        return list(required_params)
    
    def validate_parameters(self, runtime_params: Dict[str, Any]) -> bool:
        """
        验证运行时参数是否完整
        
        Args:
            runtime_params: 运行时参数
            
        Returns:
            是否验证通过
        """
        required_params = self.get_required_parameters()
        
        for param in required_params:
            if param not in runtime_params:
                print(f"Missing required parameter: {param}")
                return False
        
        return True
    
    def apply_variant(self, variant_name: str):
        """
        应用任务变体
        
        Args:
            variant_name: 变体名称
            
        Raises:
            ValueError: 变体不存在
        """
        if variant_name not in self.variants:
            raise ValueError(f"Variant '{variant_name}' not found in task config")
        
        variant = self.variants[variant_name]
        
        # 应用变体参数
        if 'parameters' in variant:
            variant_params = variant['parameters']
            
            # 更新配置中的默认参数
            if 'parameters' not in self.config:
                self.config['parameters'] = {}
            
            self.config['parameters'].update(variant_params)
    
    def get_estimated_duration(self) -> float:
        """
        估算任务执行时间
        
        Returns:
            预估时间（秒）
        """
        total_time = 0.0
        
        for state in self.states:
            timeout = state.get('timeout', 5.0)  # 默认5秒
            total_time += timeout
        
        return total_time
    
    def get_primitive_sequence(self) -> List[str]:
        """
        获取原语执行序列
        
        Returns:
            原语名称列表
        """
        return [state['primitive'] for state in self.states]
    
    def __str__(self) -> str:
        """字符串表示"""
        if not self.config:
            return "TaskConfigLoader(not loaded)"
        
        return f"TaskConfigLoader({self.task_name}, {len(self.states)} states)"
    
    def __repr__(self) -> str:
        """对象表示"""
        return self.__str__()


def load_task_config(config_path: str) -> TaskConfigLoader:
    """
    便利函数：加载任务配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        任务配置加载器实例
    """
    return TaskConfigLoader(config_path) 