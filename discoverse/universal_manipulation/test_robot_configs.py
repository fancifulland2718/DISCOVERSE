"""
机械臂配置测试工具

测试各种机械臂配置文件是否正确，通过简单的运动验证：
- 从keyframe状态开始
- 沿世界坐标系z轴向下移动0.1m
- 保持原有姿态不变
"""

import os
import time
import mujoco
import numpy as np
from pathlib import Path

from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSETS_DIR
from discoverse.universal_manipulation import UniversalTaskBase
from discoverse.universal_manipulation.robot_config import RobotConfigLoader
from discoverse.universal_manipulation.mink_solver import MinkIKSolver

class RobotConfigTester:
    """机械臂配置测试器"""
    
    def __init__(self, robot_name: str):
        """
        初始化测试器
        
        Args:
            robot_name: 机械臂名称
        """
        self.robot_name = robot_name
        self.test_results = {}
    
    def test_config_loading(self):
        """测试配置文件加载"""
        print(f"\\n🔧 测试配置文件加载...")
        try:
            config_path = project_root / "discoverse/configs/robots" / f"{self.robot_name}.yaml"
            robot_config = RobotConfigLoader(str(config_path))
            
            print(f"   ✅ 配置加载成功")
            print(f"   机械臂名称: {robot_config.robot_name}")
            print(f"   机械臂关节数: {robot_config.arm_joints}")
            print(f"   末端执行器site: {robot_config.end_effector_site}")
            print(f"   夹爪类型: {robot_config.gripper.get('type', 'N/A')}")
            
            self.test_results['config_loading'] = True
            return robot_config
            
        except Exception as e:
            print(f"   ❌ 配置加载失败: {e}")
            self.test_results['config_loading'] = False
            return None
    
    def test_mujoco_model(self, xml_path):
        """测试MuJoCo模型加载"""
        print(f"\\n🎬 测试MuJoCo模型加载...")
        try:
            model = mujoco.MjModel.from_xml_path(xml_path)
            data = mujoco.MjData(model)
            
            print(f"   ✅ 模型加载成功")
            print(f"   nq: {model.nq}, nu: {model.nu}, nkey: {model.nkey}")
            
            self.test_results['model_loading'] = True
            return model, data
            
        except Exception as e:
            print(f"   ❌ 模型加载失败: {e}")
            self.test_results['model_loading'] = False
            return None, None
    
    def test_ik_solver(self, robot_config, model, data):
        """测试IK求解器"""
        print(f"\\n🧮 测试IK求解器...")
        try:
            ik_solver = MinkIKSolver(robot_config, model, data)
            
            # 重置到keyframe状态
            mujoco.mj_resetDataKeyframe(model, data, model.key(0).id)
            mujoco.mj_forward(model, data)
            
            # 获取当前末端执行器位置和姿态
            site_name = robot_config.end_effector_site
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            current_pos = data.site_xpos[site_id].copy()
            current_ori = data.site_xmat[site_id].reshape(3, 3).copy()
            
            print(f"   当前末端位置: {current_pos}")
            print(f"   当前末端姿态矩阵:")
            for i, row in enumerate(['X轴', 'Y轴', 'Z轴']):
                print(f"      {row}: {current_ori[i]}")
            
            # 测试目标：沿z轴向下移动0.1m
            target_pos = current_pos + np.array([0, 0, -0.1])
            target_ori = current_ori  # 保持原有姿态
            
            print(f"   目标位置: {target_pos} (向下0.1m)")
            
            # 获取当前完整qpos
            current_qpos = data.qpos.copy()
            
            # 求解IK
            solution, converged, solve_info = ik_solver.solve_ik(
                target_pos, target_ori, current_qpos
            )
            
            if converged:
                print(f"   ✅ IK求解成功!")
                print(f"   位置误差: {solve_info['final_position_error']:.6f}m")
                print(f"   姿态误差: {solve_info['final_orientation_error']:.6f}rad")
                print(f"   迭代次数: {solve_info['iterations']}")
                self.test_results['ik_solver'] = True
                return solution
            else:
                print(f"   ❌ IK求解失败")
                print(f"   位置误差: {solve_info['final_position_error']:.6f}m")
                print(f"   姿态误差: {solve_info['final_orientation_error']:.6f}rad")
                print(f"   迭代次数: {solve_info['iterations']}")
                self.test_results['ik_solver'] = False
                return None
                
        except Exception as e:
            print(f"   ❌ IK求解器创建失败: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['ik_solver'] = False
            return None
    
    def test_robot_interface(self, model, data):
        """测试机械臂接口创建"""
        print(f"\\n🤖 测试机械臂接口...")
        try:
            task = UniversalTaskBase.create_from_configs(
                robot_name=self.robot_name,
                task_name="place_block",
                mj_model=model,
                mj_data=data
            )
            
            print(f"   ✅ 机械臂接口创建成功")
            print(f"   接口类型: {type(task.robot_interface).__name__}")
            print(f"   夹爪控制器: {type(task.robot_interface.gripper_controller).__name__}")
            
            self.test_results['robot_interface'] = True
            return task.robot_interface
            
        except Exception as e:
            print(f"   ❌ 机械臂接口创建失败: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['robot_interface'] = False
            return None
    
    def test_motion_execution(self, robot_interface, model, data, ik_solution):
        """测试运动执行"""
        print(f"\\n🎯 测试运动执行...")
        try:
            if ik_solution is None:
                print(f"   ⚠️ 跳过运动测试（IK求解失败）")
                self.test_results['motion_execution'] = False
                return
            
            # 重置到keyframe状态
            mujoco.mj_resetDataKeyframe(model, data, model.key(0).id)
            mujoco.mj_forward(model, data)
            
            # 记录初始位置
            site_name = robot_interface.robot_config.end_effector_site
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            initial_pos = data.site_xpos[site_id].copy()
            
            print(f"   初始位置: {initial_pos}")
            
            # 设置目标关节位置（只设置机械臂关节）
            arm_joints = len(robot_interface.robot_config.arm_joints)
            data.ctrl[:arm_joints] = ik_solution[:arm_joints]
            
            # 模拟运动（简单的步进）
            for i in range(200):  # 运行200步
                mujoco.mj_step(model, data)
            
            # 检查最终位置
            final_pos = data.site_xpos[site_id].copy()
            movement = final_pos - initial_pos
            
            print(f"   最终位置: {final_pos}")
            print(f"   实际移动: {movement}")
            print(f"   z轴移动: {movement[2]:.4f}m (目标: -0.1m)")
            
            # 验证运动是否合理（z轴移动应该接近-0.1m）
            if abs(movement[2] + 0.1) < 0.05:  # 5cm容差
                print(f"   ✅ 运动执行成功 (误差: {abs(movement[2] + 0.1)*1000:.1f}mm)")
                self.test_results['motion_execution'] = True
            else:
                print(f"   ⚠️ 运动误差较大 (误差: {abs(movement[2] + 0.1)*1000:.1f}mm)")
                self.test_results['motion_execution'] = False
                
        except Exception as e:
            print(f"   ❌ 运动执行失败: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['motion_execution'] = False
    
    def run_full_test(self):
        """运行完整测试"""
        print(f"\\n{'='*80}")
        print(f"🧪 开始测试机械臂: {self.robot_name.upper()}")
        print(f"{'='*80}")
        
        # 1. 生成模型
        xml_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf/manipulator", f"robot_{self.robot_name}.xml")
        if xml_path is None:
            print(f"\\n❌ {self.robot_name.upper()} 测试失败: 无法生成模型")
            return self.test_results
        
        # 2. 测试配置加载
        robot_config = self.test_config_loading()
        if robot_config is None:
            print(f"\\n❌ {self.robot_name.upper()} 测试失败: 配置加载失败")
            return self.test_results
        
        # 3. 测试模型加载
        model, data = self.test_mujoco_model(xml_path)
        if model is None:
            print(f"\\n❌ {self.robot_name.upper()} 测试失败: 模型加载失败")
            return self.test_results
        
        # 4. 测试IK求解器
        ik_solution = self.test_ik_solver(robot_config, model, data)
        
        # 5. 测试机械臂接口
        robot_interface = self.test_robot_interface(model, data)
        
        # 6. 测试运动执行
        if robot_interface is not None:
            self.test_motion_execution(robot_interface, model, data, ik_solution)
        
        # 汇总结果
        self.print_summary()
        return self.test_results
    
    def print_summary(self):
        """打印测试结果汇总"""
        print(f"\\n📊 {self.robot_name.upper()} 测试结果汇总:")
        print(f"{'─'*50}")
        
        test_items = [
            ('配置文件加载', 'config_loading'),
            ('MuJoCo模型', 'model_loading'),
            ('IK求解器', 'ik_solver'),
            ('机械臂接口', 'robot_interface'),
            ('运动执行', 'motion_execution')
        ]
        
        passed = 0
        total = len(test_items)
        
        for name, key in test_items:
            status = self.test_results.get(key, False)
            icon = "✅" if status else "❌"
            print(f"   {icon} {name}")
            if status:
                passed += 1
        
        print(f"{'─'*50}")
        print(f"   通过率: {passed}/{total} ({passed/total*100:.1f}%)")
        
        if passed == total:
            print(f"   🎉 {self.robot_name.upper()} 配置完全正确!")
        elif passed >= total - 1:
            print(f"   ⚠️ {self.robot_name.upper()} 配置基本正确，个别功能需要调优")
        else:
            print(f"   ❌ {self.robot_name.upper()} 配置需要修复")

def test_all_robots():
    """测试所有机械臂配置"""
    print("🚀 开始批量测试所有机械臂配置")
    print("="*80)
    
    # 获取所有机械臂配置文件
    robots_dir = project_root / "discoverse/configs/robots"
    robot_configs = []
    
    for config_file in robots_dir.glob("*.yaml"):
        robot_name = config_file.stem
        robot_configs.append(robot_name)
    
    print(f"发现 {len(robot_configs)} 个机械臂配置: {robot_configs}")
    
    # 测试结果汇总
    all_results = {}
    
    for robot_name in sorted(robot_configs):
        try:
            tester = RobotConfigTester(robot_name)
            results = tester.run_full_test()
            all_results[robot_name] = results
        except Exception as e:
            print(f"\\n❌ {robot_name.upper()} 测试过程中发生异常: {e}")
            all_results[robot_name] = {'error': str(e)}
        
        # 添加间隔
        time.sleep(1)
    
    # 打印最终汇总
    print("\\n" + "="*80)
    print("🏆 所有机械臂测试结果汇总")
    print("="*80)
    
    for robot_name, results in all_results.items():
        if 'error' in results:
            print(f"❌ {robot_name.upper()}: 测试异常")
        else:
            passed = sum(1 for v in results.values() if v)
            total = len(results)
            if passed == total:
                print(f"✅ {robot_name.upper()}: 完全通过 ({passed}/{total})")
            elif passed >= total - 1:
                print(f"⚠️ {robot_name.upper()}: 基本通过 ({passed}/{total})")
            else:
                print(f"❌ {robot_name.upper()}: 需要修复 ({passed}/{total})")

def test_single_robot(robot_name: str):
    """测试单个机械臂"""
    tester = RobotConfigTester(robot_name)
    return tester.run_full_test()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="机械臂配置测试工具")
    parser.add_argument("-r", "--robot", type=str, help="测试指定机械臂 (不指定则测试所有)")
    parser.add_argument("--list", action="store_true", help="列出所有可用的机械臂配置")
    
    args = parser.parse_args()
    
    project_root = Path(DISCOVERSE_ROOT_DIR)
    if args.list:
        robots_dir = project_root / "discoverse/configs/robots"
        robot_configs = [f.stem for f in robots_dir.glob("*.yaml")]
        print("可用的机械臂配置:")
        for robot in sorted(robot_configs):
            print(f"  - {robot}")
    elif args.robot:
        test_single_robot(args.robot)
    else:
        test_all_robots()
