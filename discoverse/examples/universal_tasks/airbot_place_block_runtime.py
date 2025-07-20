"""
AirBot Play机械臂place_block任务演示 - [运行架构]版本

采用原始place_block.py的高效运行架构：
- 高频主循环 (物理模拟240Hz)
- 低频任务设置 (非阻塞)
- 平滑控制执行
- 终止条件检查

同时保留universal_manipulati                # 求解IK
                solution, converged, solve_info = self.task.robot_interface.ik_solver.solve_ik(
                    target_pos, target_ori, full_current_qpos
                )
                
                if converged:
                    # IK求解器返回8个值，但MuJoCo只有7个控制器（6臂+1夹爪）
                    # 只取前6个机械臂关节
                    self.target_control[:self.arm_joints] = solution[:self.arm_joints]
                    print(f"   ✅ 目标移动IK成功: 位置误差 {solve_info['final_position_error']:.6f}")
                else:
                    print(f"   ❌ 目标移动IK失败: 位置误差 {solve_info['final_position_error']:.6f}")
                    return False配置驱动系统
- 动作原语
- Mink IK求解器
"""

import sys
import time
import numpy as np
import mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

print("🎬 强制启用可视化模式 (运行架构版)")

from discoverse import DISCOVERSE_ASSETS_DIR
from discoverse.universal_manipulation import UniversalTaskBase
from discoverse.utils import SimpleStateMachine, step_func, get_body_tmat

class RuntimeTaskExecutor:
    """运行时任务执行器 - 采用高频循环架构"""
    
    def __init__(self, task, viewer, model, data):
        """
        初始化运行时执行器
        
        Args:
            task: UniversalTaskBase任务实例
            viewer: MuJoCo viewer
            model: MuJoCo模型
            data: MuJoCo数据
        """
        self.task = task
        self.viewer = viewer
        self.model = model
        self.data = data
        
        # 任务配置
        self.resolved_states = task.task_config.get_resolved_states()
        self.total_states = len(self.resolved_states)
        
        # 状态机
        self.stm = SimpleStateMachine()
        self.stm.max_state_cnt = self.total_states
        
        # 控制状态 - 使用MuJoCo实际控制器数量 (7个: 6臂+1夹爪)
        self.mujoco_ctrl_dim = model.nu  # MuJoCo控制器维度
        self.target_control = np.zeros(self.mujoco_ctrl_dim)
        self.action = np.zeros(self.mujoco_ctrl_dim)
        self.move_speed = 0.75  # 控制速度
        self.joint_move_ratio = np.ones(self.mujoco_ctrl_dim)
        
        # 运行时状态
        self.running = True
        self.max_time = 30.0  # 最大执行时间
        self.start_time = time.time()
        self.success = False
        
        # 从任务配置获取机械臂维度信息
        self.arm_joints = len(task.robot_interface.arm_joints)  # 机械臂关节数
        self.gripper_ctrl_idx = self.arm_joints  # 夹爪控制索引在机械臂关节之后
        
        # 初始化动作
        self.action[:] = self.get_current_qpos()[:self.mujoco_ctrl_dim]
        
        print(f"🤖 运行时执行器初始化完成")
        print(f"   总状态数: {self.total_states}")
        print(f"   机械臂自由度: {self.arm_joints}")
        print(f"   MuJoCo控制器维度: {self.mujoco_ctrl_dim}")
        print(f"   夹爪控制索引: {self.gripper_ctrl_idx}")
    
    def get_current_qpos(self):
        """获取当前关节位置"""
        return self.data.qpos.copy()
    
    def check_action_done(self):
        """检查动作是否完成"""
        current_qpos = self.get_current_qpos()
        # 只检查前6个机械臂关节
        position_error = np.linalg.norm(current_qpos[:self.arm_joints] - self.target_control[:self.arm_joints])
        return position_error < 0.02  # 2cm容差
    
    def set_target_from_primitive(self, state_config):
        """使用原语设置目标控制信号"""
        try:
            primitive = state_config["primitive"]
            params = state_config.get("params", {})
            gripper_state = state_config.get("gripper_state", "open")
            
            print(f"   🔧 执行原语: {primitive}")
            
            if primitive == "move_to_object":
                # 使用原语计算目标位置
                object_name = params.get("object_name", "")
                offset = np.array(params.get("offset", [0, 0, 0]))
                
                if object_name:
                    # 获取物体位置
                    object_tmat = get_body_tmat(self.data, object_name)
                    target_pos = object_tmat[:3, 3] + offset
                    
                    # 获取当前末端执行器姿态矩阵（从MuJoCo数据直接读取）
                    site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "endpoint")
                    current_ori = self.data.site_xmat[site_id].reshape(3, 3).copy()
                    
                    print(f"   🤖 当前末端姿态矩阵:\n{current_ori}")
                    print(f"   🎯 目标位置: {target_pos}")
                    print(f"   ✅ 使用当前姿态作为目标（避免大幅度旋转）")
                    
                    # 获取完整的qpos (包含所有自由度)
                    full_current_qpos = self.data.qpos.copy()
                    
                    # 求解IK
                    solution, converged, solve_info = self.task.robot_interface.ik_solver.solve_ik(
                        target_pos, current_ori, full_current_qpos
                    )
                    
                    if converged:
                        # Mink IK求解器返回前6维作为机械臂关节控制
                        self.target_control[:self.arm_joints] = solution[:self.arm_joints]
                        print(f"   ✅ IK求解成功: 误差 {solve_info['final_position_error']:.6f}")
                    else:
                        print(f"   ❌ IK求解失败: 误差 {solve_info['final_position_error']:.6f}")
                        return False
                        
            elif primitive == "move_relative":
                # 相对移动
                offset = np.array(params.get("offset", [0, 0, 0]))
                
                # 获取当前位置
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "endpoint")
                current_pos = self.data.site_xpos[site_id].copy()
                current_ori = self.data.site_xmat[site_id].reshape(3, 3).copy()
                
                target_pos = current_pos + offset
                
                print(f"   🤖 当前位置: {current_pos}")
                print(f"   🎯 目标位置: {target_pos} (偏移: {offset})")
                
                # 获取完整的qpos
                full_current_qpos = self.data.qpos.copy()
                
                # 求解IK
                solution, converged, solve_info = self.task.robot_interface.ik_solver.solve_ik(
                    target_pos, current_ori, full_current_qpos
                )
                
                if converged:
                    # Mink IK求解器返回前6维作为机械臂关节控制
                    self.target_control[:self.arm_joints] = solution[:self.arm_joints]
                    print(f"   ✅ 相对移动IK成功: {offset}, 误差 {solve_info['final_position_error']:.6f}")
                else:
                    print(f"   ❌ 相对移动IK失败: 误差 {solve_info['final_position_error']:.6f}")
                    return False
            
            elif primitive in ["grasp_object", "release_object", "set_gripper"]:
                # 夹爪控制 - 不需要IK，直接设置夹爪状态
                print(f"   🤏 夹爪控制: {gripper_state}")
            
            # 设置夹爪状态
            if gripper_state == "open":
                self.target_control[self.gripper_ctrl_idx] = 1.0  # 打开夹爪
            elif gripper_state == "close":
                self.target_control[self.gripper_ctrl_idx] = 0.0  # 关闭夹爪
            
            # 计算关节移动比例（用于速度控制）
            current_ctrl = self.data.ctrl[:self.mujoco_ctrl_dim].copy()
            dif = np.abs(current_ctrl - self.target_control)
            self.joint_move_ratio = dif / (np.max(dif) + 1e-6)
            
            return True
            
        except Exception as e:
            print(f"   ❌ 原语执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step(self):
        """单步执行 - 高频主循环"""
        try:
            # 状态机触发检查 (低频)
            if self.stm.trigger():
                if self.stm.state_idx < self.total_states:
                    state_config = self.resolved_states[self.stm.state_idx]
                    print(f"\n🎯 状态 {self.stm.state_idx+1}/{self.total_states}: {state_config['name']}")
                    
                    # 设置mocap可视化
                    self.set_mocap_target(state_config)
                    
                    # 使用原语设置目标
                    if not self.set_target_from_primitive(state_config):
                        print(f"   ❌ 状态 {self.stm.state_idx} 设置失败")
                        return False
                else:
                    # 所有状态完成，检查成功条件
                    self.success = self.check_task_success()
                    self.running = False
                    return True
                    
            # 超时检查
            elif time.time() - self.start_time > self.max_time:
                print("❌ 任务超时")
                self.running = False
                return False
            else:
                # 更新状态机
                self.stm.update()
            
            # 检查动作完成条件 (高频)
            if self.check_action_done():
                print(f"   ✅ 状态 {self.stm.state_idx+1} 完成")
                self.stm.next()
            
            # 平滑控制执行 (高频) - 只控制前6个机械臂关节
            for i in range(self.arm_joints):
                self.action[i] = step_func(
                    self.action[i], 
                    self.target_control[i], 
                    self.move_speed * self.joint_move_ratio[i] * (1/240)  # 假设240Hz
                )
            # 夹爪直接设置
            self.action[self.gripper_ctrl_idx] = self.target_control[self.gripper_ctrl_idx]
            
            # 设置控制信号到MuJoCo - 使用实际控制器维度
            self.data.ctrl[:self.mujoco_ctrl_dim] = self.action[:self.mujoco_ctrl_dim]
            
            # 物理步进 (高频)
            mujoco.mj_step(self.model, self.data)
            
            # 可视化同步
            if self.viewer is not None:
                self.viewer.sync()
            
            return True
            
        except Exception as e:
            print(f"❌ 步进失败: {e}")
            self.running = False
            return False
    
    def set_mocap_target(self, state_config):
        """设置mocap目标可视化"""
        try:
            if 'move_to_object' in state_config.get('primitive', ''):
                object_name = state_config.get('params', {}).get('object_name', '')
                offset = state_config.get('params', {}).get('offset', [0, 0, 0])
                
                if object_name and hasattr(self.data, 'body'):
                    object_pos = self.data.body(object_name).xpos.copy()
                    target_pos = object_pos + np.array(offset)
                    
                    # 设置mocap目标位置
                    mocap_id = self.model.body('target').mocapid
                    if mocap_id >= 0:
                        self.data.mocap_pos[mocap_id] = target_pos
                        self.model.geom('target_box').rgba = np.array([1.0, 1.0, 0.3, 0.3])  # 黄色目标
                        print(f"   🎯 Mocap目标: {target_pos}")
        except Exception as e:
            print(f"   ⚠️ Mocap设置失败: {e}")
    
    def check_task_success(self):
        """检查任务成功条件"""
        try:
            # 检查绿色方块是否在粉色碗中
            block_pos = self.data.body('block_green').xpos
            bowl_pos = self.data.body('bowl_pink').xpos
            distance = np.linalg.norm(block_pos[:2] - bowl_pos[:2])  # 只检查XY平面
            return distance < 0.03  # 3cm容差
        except:
            return False
    
    def run(self):
        """运行任务主循环"""
        print(f"\n🚀 开始运行时执行 (运行架构版)")
        print(f"   高频物理循环 + 低频状态切换")
        print(f"   最大时间: {self.max_time}s")
        
        step_count = 0
        last_report_time = time.time()
        
        while self.running:
            if not self.step():
                break
                
            step_count += 1
            
            # 每秒报告一次进度
            if time.time() - last_report_time > 1.0:
                elapsed = time.time() - self.start_time
                print(f"   ⏱️  运行时间: {elapsed:.1f}s, 步数: {step_count}, 当前状态: {self.stm.state_idx+1}/{self.total_states}")
                last_report_time = time.time()
        
        # 报告结果
        elapsed_time = time.time() - self.start_time
        print(f"\n📊 运行架构执行完成!")
        print(f"   总时间: {elapsed_time:.2f}s")
        print(f"   总步数: {step_count}")
        print(f"   完成状态: {self.stm.state_idx}/{self.total_states}")
        print(f"   任务成功: {'✅ 是' if self.success else '❌ 否'}")
        
        return self.success

def generate_airbot_place_block_model():
    """生成AirBot Play place_block模型"""
    sys.path.insert(0, str(project_root / "discoverse/envs"))
    from make_env import make_env
    
    xml_path = "airbot_place_block_mink.xml"
    env = make_env("airbot_play", "place_block", xml_path)
    print(f"🏗️ 生成AirBot Play模型: {xml_path}")
    return xml_path

def setup_scene(model, data):
    """初始化场景"""
    # 重置到home位置
    mujoco.mj_resetDataKeyframe(model, data, model.key(0).id)
    mujoco.mj_forward(model, data)
    
    # 初始化mocap target
    try:
        import mink
        mink.move_mocap_to_frame(model, data, "target", "endpoint", "site")
        print("🎯 Mocap target初始化成功")
    except Exception as e:
        print(f"⚠️ Mocap初始化失败: {e}")
    
    print("🎬 场景初始化完成")
    print(f"   绿色方块位置: {data.body('block_green').xpos}")
    print(f"   粉色碗位置: {data.body('bowl_pink').xpos}")
    print(f"   机械臂末端位置: {data.site('endpoint').xpos}")

def create_simple_visualizer(model, data):
    """创建MuJoCo内置可视化器"""
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    print("🎬 MuJoCo内置查看器创建成功")
    return viewer

def main():
    """主函数 - 运行架构版"""
    print("🤖 启动AirBot Play place_block任务演示 (运行架构版)")
    print("=" * 70)
    
    # 生成模型
    try:
        xml_path = generate_airbot_place_block_model()
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print(f"✅ 模型加载成功! (nq={model.nq}, nkey={model.nkey})")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 初始化场景
    setup_scene(model, data)

    # 创建查看器
    viewer = create_simple_visualizer(model, data)

    # 创建通用任务
    try:
        task = UniversalTaskBase.create_from_configs(
            robot_name="airbot_play",
            task_name="place_block",
            mj_model=model,
            mj_data=data
        )
        print(f"✅ 任务创建成功")
        
        # 设置viewer引用
        task.robot_interface.set_viewer(viewer)
        
    except Exception as e:
        print(f"❌ 任务创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建运行时执行器
    try:
        executor = RuntimeTaskExecutor(task, viewer, model, data)
        
        # 运行任务
        success = executor.run()
        
        if success:
            print(f"\n🎉 运行架构任务成功完成!")
            print(f"   绿色方块已成功放入粉色碗中")
        else:
            print(f"\n⚠️ 运行架构任务未完全成功")
        
    except Exception as e:
        print(f"❌ 运行时执行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭查看器
        if viewer is not None:
            try:
                viewer.close()
                print("🎬 查看器已关闭")
            except:
                pass

if __name__ == "__main__":
    main()
