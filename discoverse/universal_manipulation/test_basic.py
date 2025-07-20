"""
基础架构测试

验证配置加载器、Mink求解器和原语注册器的基本功能。
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_robot_config_loader():
    """测试机械臂配置加载器"""
    print("=== 测试机械臂配置加载器 ===")
    
    try:
        from discoverse.universal_manipulation import RobotConfigLoader
        
        # 测试Panda配置
        panda_config_path = project_root / "discoverse" / "configs" / "robots" / "panda.yaml"
        if panda_config_path.exists():
            panda_config = RobotConfigLoader(str(panda_config_path))
            print(f"✅ 成功加载Panda配置: {panda_config}")
            print(f"   - 机械臂名称: {panda_config.robot_name}")
            print(f"   - 自由度: {panda_config.dof}")
            print(f"   - 机械臂关节数: {panda_config.arm_joints}")
            print(f"   - 末端执行器: {panda_config.end_effector_site}")
        else:
            print("❌ Panda配置文件不存在")
        
        # 测试AirBot Play配置
        airbot_config_path = project_root / "discoverse" / "configs" / "robots" / "airbot_play.yaml"
        if airbot_config_path.exists():
            airbot_config = RobotConfigLoader(str(airbot_config_path))
            print(f"✅ 成功加载AirBot Play配置: {airbot_config}")
            print(f"   - 机械臂名称: {airbot_config.robot_name}")
            print(f"   - 自由度: {airbot_config.dof}")
            print(f"   - 机械臂关节数: {airbot_config.arm_joints}")
        else:
            print("❌ AirBot Play配置文件不存在")
            
    except Exception as e:
        print(f"❌ 配置加载器测试失败: {e}")

def test_task_config_loader():
    """测试任务配置加载器"""
    print("\n=== 测试任务配置加载器 ===")
    
    try:
        from discoverse.universal_manipulation import TaskConfigLoader
        
        # 测试放置物体任务
        task_config_path = project_root / "discoverse" / "configs" / "tasks" / "place_object.yaml"
        if task_config_path.exists():
            task_config = TaskConfigLoader(str(task_config_path))
            print(f"✅ 成功加载任务配置: {task_config}")
            print(f"   - 任务名称: {task_config.task_name}")
            print(f"   - 状态数量: {len(task_config.states)}")
            print(f"   - 必需参数: {task_config.get_required_parameters()}")
            
            # 测试参数替换
            task_config.set_runtime_parameters(
                source_object="block_green",
                target_location="bowl_pink"
            )
            resolved_states = task_config.get_resolved_states()
            print(f"   - 参数替换测试: 第一个状态的物体名称: {resolved_states[0]['params']['object_name']}")
        else:
            print("❌ 任务配置文件不存在")
            
    except Exception as e:
        print(f"❌ 任务配置加载器测试失败: {e}")

def test_primitive_registry():
    """测试原语注册器"""
    print("\n=== 测试原语注册器 ===")
    
    try:
        from discoverse.universal_manipulation.primitives import PrimitiveRegistry
        
        # 创建注册器
        registry = PrimitiveRegistry()
        print(f"✅ 成功创建原语注册器: {registry}")
        
        # 列出所有原语
        primitives = registry.list_primitives()
        print(f"   - 注册的原语数量: {len(primitives)}")
        print(f"   - 原语列表: {primitives}")
        
        # 测试获取原语信息
        if "move_to_object" in primitives:
            info = registry.get_primitive_info("move_to_object")
            print(f"   - move_to_object信息: {info['description']}")
            print(f"   - 必需参数: {info['required_parameters']}")
        
        # 测试参数验证
        if "move_relative" in primitives:
            valid, msg = registry.validate_primitive_params("move_relative", {"offset": [0, 0, 0.1]})
            print(f"   - move_relative参数验证: {valid} - {msg}")
            
    except Exception as e:
        print(f"❌ 原语注册器测试失败: {e}")

def test_mink_solver():
    """测试Mink求解器"""
    print("\n=== 测试Mink求解器 ===")
    
    try:
        # 检查mink是否可用
        try:
            import mink
            print("✅ Mink库可用")
        except ImportError:
            print("⚠️  Mink库未安装，跳过Mink测试")
            return
        
        # 这里需要实际的MuJoCo模型才能测试
        # 暂时只检查是否能导入
        from discoverse.universal_manipulation import MinkIKSolver
        print("✅ MinkIKSolver类可以正常导入")
        
    except Exception as e:
        print(f"❌ Mink求解器测试失败: {e}")

def test_config_files():
    """测试配置文件格式"""
    print("\n=== 测试配置文件格式 ===")
    
    try:
        import yaml
        
        # 测试原语配置文件
        primitives_config_path = project_root / "discoverse" / "configs" / "primitives" / "manipulation_primitives.yaml"
        if primitives_config_path.exists():
            with open(primitives_config_path, 'r', encoding='utf-8') as f:
                primitives_config = yaml.safe_load(f)
            print(f"✅ 原语配置文件格式正确，包含{len(primitives_config.get('primitives', {}))}个原语定义")
        else:
            print("❌ 原语配置文件不存在")
        
        # 测试机械臂配置文件
        configs_dir = project_root / "discoverse" / "configs" / "robots"
        robot_configs = list(configs_dir.glob("*.yaml"))
        print(f"✅ 发现{len(robot_configs)}个机械臂配置文件")
        
        # 测试任务配置文件
        task_configs_dir = project_root / "discoverse" / "configs" / "tasks"
        task_configs = list(task_configs_dir.glob("*.yaml"))
        print(f"✅ 发现{len(task_configs)}个任务配置文件")
        
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")

def main():
    """运行所有测试"""
    print("🚀 开始Universal Manipulation基础架构测试\n")
    
    test_config_files()
    test_robot_config_loader()
    test_task_config_loader()
    test_primitive_registry()
    test_mink_solver()
    
    print("\n🎉 基础架构测试完成！")

if __name__ == "__main__":
    main() 