#!/usr/bin/env python3
"""
Universal Task Runtime - 改进版本

集成了以下改进：
1. 使用统一的utils模块
2. 简化的错误处理
3. 模板化配置支持
4. CICD测试集成
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path

import mink
import mujoco
import numpy as np
import yaml

import discoverse
from discoverse.envs import make_env
from discoverse import DISCOVERSE_ASSETS_DIR
from discoverse.universal_manipulation import UniversalTaskBase
from discoverse.universal_manipulation.utils import (
    SimpleStateMachine, step_func, get_body_tmat,
    validate_mujoco_object, calculate_distance
)


def load_and_resolve_config(config_path: str) -> dict:
    """加载并解析配置文件（支持模板继承）
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        解析后的配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查是否有模板继承
    if 'extends' in config:
        template_path = config['extends']
        if not os.path.isabs(template_path):
            # 相对路径，相对于当前配置文件
            base_dir = os.path.dirname(config_path)
            template_path = os.path.join(base_dir, template_path)
        
        print(f"📄 加载模板: {template_path}")
        
        # 递归加载模板
        template_config = load_and_resolve_config(template_path)
        
        # 合并配置（当前配置覆盖模板）
        merged_config = merge_configs(template_config, config)
        return merged_config
    
    return config


def merge_configs(template: dict, override: dict) -> dict:
    """合并配置文件（深度合并，支持状态数组的智能合并）
    
    Args:
        template: 模板配置
        override: 覆盖配置
        
    Returns:
        合并后的配置
    """
    result = template.copy()
    
    for key, value in override.items():
        if key == 'extends':
            continue  # 跳过extends字段
            
        if key == 'states' and isinstance(result.get(key), list) and isinstance(value, list):
            # 特殊处理states数组：按索引合并
            result[key] = merge_states_array(result[key], value)
        elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def merge_states_array(template_states: list, override_states: list) -> list:
    """合并状态数组（按索引覆盖）
    
    Args:
        template_states: 模板中的状态数组
        override_states: 覆盖配置中的状态数组
        
    Returns:
        合并后的状态数组
    """
    # 从模板开始
    result = template_states.copy()
    
    # 按索引覆盖
    for i, override_state in enumerate(override_states):
        if i < len(result):
            # 覆盖已有的状态
            result[i] = override_state
        else:
            # 添加新状态
            result.append(override_state)
    
    return result


def replace_variables(config: dict) -> dict:
    """替换配置中的变量引用
    
    Args:
        config: 原始配置
        
    Returns:
        替换变量后的配置
    """
    import re
    import json
    
    # 获取运行时参数
    runtime_params = config.get('runtime_parameters', {})
    
    # 将配置转换为JSON字符串进行替换
    config_str = json.dumps(config, ensure_ascii=False)
    
    # 替换${variable}格式的变量
    for key, value in runtime_params.items():
        # 1. 替换带引号的变量（保持数据类型）
        quoted_pattern = f"\"${{{key}}}\""
        if isinstance(value, (int, float)):
            quoted_replacement = str(value)  # 数值不加引号
        else:
            quoted_replacement = f'"{value}"'  # 字符串加引号
        config_str = config_str.replace(quoted_pattern, quoted_replacement)
        
        # 2. 替换字符串内的变量（如description中的变量）
        inline_pattern = f"${{{key}}}"
        inline_replacement = str(value)  # 都转为字符串
        config_str = config_str.replace(inline_pattern, inline_replacement)
    
    # 转换回字典
    return json.loads(config_str)


def validate_task_config(config: dict, robot_name: str) -> bool:
    """验证任务配置
    
    Args:
        config: 任务配置
        robot_name: 机械臂名称
        
    Returns:
        是否有效
    """
    # 检查必需字段（兼容新旧格式）
    required_fields = ['task_name', 'description']
    
    # 检查状态字段（支持states或task_states）
    if 'states' not in config and 'task_states' not in config:
        print(f"❌ 配置缺少状态字段: states 或 task_states")
        return False
    
    for field in required_fields:
        if field not in config:
            print(f"❌ 配置缺少必需字段: {field}")
            return False
    
    # 检查机械臂兼容性
    if 'task_meta' in config and 'compatibility' in config['task_meta']:
        compatible_robots = config['task_meta']['compatibility'].get('robots', [])
        if robot_name not in compatible_robots:
            print(f"⚠️ 机械臂 {robot_name} 可能不兼容此任务")
            print(f"   支持的机械臂: {compatible_robots}")
    
    return True

class UniversalRuntimeTaskExecutor:
    """通用运行时任务执行器 - 改进版本
    
    集成了utils模块、简化的错误处理、模板化配置支持
    """

    def __init__(self, task: UniversalTaskBase, viewer, mj_model: mujoco.MjModel, 
                 mj_data: mujoco.MjData, robot_name: str, sync: bool = False):
        """初始化运行时执行器
        
        Args:
            task: UniversalTaskBase任务实例
            viewer: MuJoCo viewer
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
            robot_name: 机械臂名称
            sync: 是否启用实时同步
        """
        self.task = task
        self.viewer = viewer
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.robot_name = robot_name
        self.sync = sync
        
        # 时间和频率控制
        self.sim_timestep = mj_model.opt.timestep
        self.render_fps = 60
        
        # 任务配置 - 支持模板化配置
        try:
            self.resolved_states = task.task_config.get_resolved_states()
            self.total_states = len(self.resolved_states)
        except Exception as e:
            print(f"❌ 任务配置解析失败: {e}")
            raise
        
        # 状态机
        self.stm = SimpleStateMachine()
        self.stm.max_state_cnt = self.total_states
        
        # 控制状态
        self.mujoco_ctrl_dim = mj_model.nu
        self.target_control = np.zeros(self.mujoco_ctrl_dim)
        self.action = np.zeros(self.mujoco_ctrl_dim)
        self.move_speed = 0.75  # 控制速度
        self.joint_move_ratio = np.ones(self.mujoco_ctrl_dim)
        
        # 运行时状态
        self.running = True
        self.max_time = 30.0  # 最大执行时间
        self.start_time = time.time()
        self.success = False
        self.viewer_closed = False  # 新增: 标记viewer是否被关闭
        
        # 延时支持
        self.current_delay = 0.0  # 当前状态的延时时间
        self.delay_start_sim_time = None  # 延时开始的仿真时间
        
        # 从任务配置获取机械臂维度信息
        self.arm_joints = len(task.robot_interface.arm_joints)  # 机械臂关节数
        self.gripper_ctrl_idx = self.arm_joints  # 夹爪控制索引在机械臂关节之后
        
        # 初始化动作
        self.action[:] = self.get_current_qpos()[:self.mujoco_ctrl_dim]
        
        print(f"🤖 {robot_name.upper()} 运行时执行器初始化完成")
        print(f"   总状态数: {self.total_states}")
        print(f"   机械臂自由度: {self.arm_joints}")
        print(f"   MuJoCo控制器维度: {self.mujoco_ctrl_dim}")
        print(f"   夹爪控制索引: {self.gripper_ctrl_idx}")
        print(f"   实时同步: {'✅ 启用' if self.sync else '❌ 禁用'}")
        print(f"   渲染频率: {self.render_fps} Hz")
        print(f"   仿真时间步长: {self.sim_timestep} s")
    
    def get_current_qpos(self):
        """获取当前关节位置"""
        return self.mj_data.qpos.copy()
    
    def check_action_done(self):
        """检查动作是否完成"""
        current_qpos = self.get_current_qpos()
        # 只检查机械臂关节
        position_error = np.linalg.norm(current_qpos[:self.arm_joints] - self.target_control[:self.arm_joints])
        position_done = position_error < 0.02  # 2cm容差
        
        # 检查延时条件
        if self.current_delay > 0 and self.delay_start_sim_time is not None:
            delay_elapsed = self.mj_data.time - self.delay_start_sim_time
            delay_done = delay_elapsed >= self.current_delay
            if not delay_done:
                return False  # 延时未完成，动作未完成
            
        return position_done
    
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
                offset = params.get("offset", [0, 0, 0])
                
                # 确保offset是数字数组
                if isinstance(offset, list):
                    try:
                        offset = np.array([float(x) for x in offset])
                    except:
                        print(f"   ❌ offset转换失败: {offset}")
                        return False
                else:
                    offset = np.array(offset)
                
                if object_name:
                    # 获取物体位置
                    object_tmat = get_body_tmat(self.mj_data, object_name)
                    target_pos = object_tmat[:3, 3] + offset
                    
                    # 获取当前末端执行器姿态矩阵（从MuJoCo数据直接读取）
                    site_name = self.task.robot_interface.robot_config.end_effector_site
                    site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                    current_ori = self.mj_data.site_xmat[site_id].reshape(3, 3).copy()
                    
                    print(f"   🎯 目标位置: {target_pos}")
                    print(f"   ✅ 使用当前姿态作为目标（避免大幅度旋转）")
                    
                    # 获取完整的qpos (包含所有自由度)
                    full_current_qpos = self.mj_data.qpos.copy()
                    
                    # 求解IK
                    solution, converged, solve_info = self.task.robot_interface.ik_solver.solve_ik(
                        target_pos, current_ori, full_current_qpos
                    )
                    
                    if converged:
                        # IK求解器返回机械臂关节解
                        self.target_control[:self.arm_joints] = solution[:self.arm_joints]
                        print(f"   ✅ IK求解成功: 误差 {solve_info['final_position_error']:.6f}")
                    else:
                        print(f"   ❌ IK求解失败: 误差 {solve_info['final_position_error']:.6f}")
                        return False
                        
            elif primitive == "move_relative":
                # 相对移动
                offset = params.get("offset", [0, 0, 0])
                
                # 确保offset是数字数组
                if isinstance(offset, list):
                    try:
                        offset = np.array([float(x) for x in offset])
                    except:
                        print(f"   ❌ offset转换失败: {offset}")
                        return False
                else:
                    offset = np.array(offset)
                
                # 获取当前位置
                site_name = self.task.robot_interface.robot_config.end_effector_site
                site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                current_pos = self.mj_data.site_xpos[site_id].copy()
                current_ori = self.mj_data.site_xmat[site_id].reshape(3, 3).copy()
                
                target_pos = current_pos + offset
                
                print(f"   🤖 当前位置: {current_pos}")
                print(f"   🎯 目标位置: {target_pos} (偏移: {offset})")
                
                # 获取完整的qpos
                full_current_qpos = self.mj_data.qpos.copy()
                
                # 求解IK
                solution, converged, solve_info = self.task.robot_interface.ik_solver.solve_ik(
                    target_pos, current_ori, full_current_qpos
                )
                
                if converged:
                    # IK求解器返回机械臂关节解
                    self.target_control[:self.arm_joints] = solution[:self.arm_joints]
                    print(f"   ✅ 相对移动IK成功: {offset}, 误差 {solve_info['final_position_error']:.6f}")
                else:
                    print(f"   ❌ 相对移动IK失败: 误差 {solve_info['final_position_error']:.6f}")
                    return False
            
            elif primitive in ["grasp_object", "release_object", "set_gripper"]:
                # 夹爪控制 - 不需要IK，直接设置夹爪状态
                print(f"   🤏 夹爪控制: {gripper_state}")
            
            # 设置夹爪状态 - 使用夹爪控制器
            if gripper_state == "open":
                self.target_control[self.gripper_ctrl_idx] = self.task.robot_interface.gripper_controller.open()
            elif gripper_state == "close":
                self.target_control[self.gripper_ctrl_idx] = self.task.robot_interface.gripper_controller.close()
            
            # 计算关节移动比例（用于速度控制）
            current_ctrl = self.mj_data.ctrl[:self.mujoco_ctrl_dim].copy()
            dif = np.abs(current_ctrl - self.target_control)
            self.joint_move_ratio = dif / (np.max(dif) + 1e-6)
            
            return True
            
        except Exception as e:
            print(f"   ❌ 原语执行失败: {e}")
            traceback.print_exc()
            return False
    
    def step(self):
        """单步执行 - 高频主循环"""
        try:
            # 状态机触发检查 (低频)
            if self.stm.trigger():
                if self.stm.state_idx < self.total_states:
                    state_config = self.resolved_states[self.stm.state_idx]
                    print(f"\\n🎯 状态 {self.stm.state_idx+1}/{self.total_states}: {state_config['name']}")
                    
                    # 获取延时配置
                    self.current_delay = state_config.get("delay", 0.0)
                    if isinstance(self.current_delay, str):
                        try:
                            self.current_delay = float(self.current_delay)
                        except:
                            self.current_delay = 0.0
                            
                    if self.current_delay > 0:
                        print(f"   ⏱️  状态延时: {self.current_delay}s")
                    
                    # 设置mocap可视化
                    self.set_mocap_target(state_config)
                    
                    # 使用原语设置目标
                    if not self.set_target_from_primitive(state_config):
                        print(f"   ❌ 状态 {self.stm.state_idx} 设置失败")
                        return False
                        
                    # 如果有延时，记录开始的仿真时间
                    if self.current_delay > 0:
                        self.delay_start_sim_time = self.mj_data.time
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
                # 如果有延时，显示延时完成信息
                if self.current_delay > 0 and self.delay_start_sim_time is not None:
                    delay_elapsed = self.mj_data.time - self.delay_start_sim_time
                    print(f"   ⏱️  延时完成: {delay_elapsed:.2f}s / {self.current_delay}s (仿真时间)")
                
                print(f"   ✅ 状态 {self.stm.state_idx+1} 完成")
                
                # 重置延时相关变量
                self.current_delay = 0.0
                self.delay_start_sim_time = None
                
                self.stm.next()
            
            # 平滑控制执行 (高频) - 只控制机械臂关节
            for i in range(self.arm_joints):
                self.action[i] = step_func(
                    self.action[i], 
                    self.target_control[i], 
                    self.move_speed * self.joint_move_ratio[i] * self.mj_model.opt.timestep
                )
            # 夹爪直接设置
            self.action[self.gripper_ctrl_idx] = self.target_control[self.gripper_ctrl_idx]
            
            # 设置控制信号到MuJoCo - 使用实际控制器维度
            self.mj_data.ctrl[:self.mujoco_ctrl_dim] = self.action[:self.mujoco_ctrl_dim]
            
            # 物理步进 (高频)
            mujoco.mj_step(self.mj_model, self.mj_data)

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
                
                if object_name and hasattr(self.mj_data, 'body'):
                    object_pos = self.mj_data.body(object_name).xpos.copy()
                    target_pos = object_pos + np.array(offset)
                    
                    # 设置mocap目标位置
                    try:
                        mocap_id = self.mj_model.body('target').mocapid
                        if mocap_id >= 0:
                            self.mj_data.mocap_pos[mocap_id] = target_pos
                            self.mj_model.geom('target_box').rgba = np.array([1.0, 1.0, 0.3, 0.3])  # 黄色目标
                            print(f"   🎯 Mocap目标: {target_pos}")
                    except:
                        pass  # 如果没有mocap目标，忽略
        except Exception as e:
            print(f"   ⚠️ Mocap设置失败: {e}")
    
    def check_task_success(self):
        """检查任务成功条件 - 简化版本"""
        print(f"\\n🔍 开始任务成功检查...")
        
        # 使用统一的成功检查接口
        try:
            success = self.task.check_success()
            status = "✅ 通过" if success else "❌ 未通过"
            print(f"   {status}")
            return success
        except Exception as e:
            print(f"   💥 检查异常: {e}")
            return False
    
    def run(self):
        """运行任务主循环"""
        sync_mode = "实时同步" if self.sync else "高速执行"
        print(f"\\n🚀 开始{self.robot_name.upper()}运行时执行 (通用运行架构版)")
        print(f"   高频物理循环 + 低频状态切换")
        print(f"   最大时间: {self.max_time}s")
        print(f"   执行模式: {sync_mode}")
        
        step_count = 0
        last_report_time = time.time()
        
        # 实时同步相关变量
        if self.sync:
            real_start_time = time.time()
            expected_sim_time = 0.0
        
        last_render_time = 0.0
        
        while self.running:
            if not self.step():
                break
                
            step_count += 1

            # 实时同步控制
            if self.sync:
                expected_sim_time = self.mj_data.time
                real_elapsed = time.time() - real_start_time
                sim_elapsed = expected_sim_time
                
                # 如果仿真跑得太快，等待实际时间追上
                if sim_elapsed > real_elapsed:
                    sleep_time = sim_elapsed - real_elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            # 检查viewer是否被关闭 - 使用官方API
            if self.viewer is not None:
                if not self.viewer.is_running():
                    print("🎬 查看器已关闭，退出程序")
                    self.viewer_closed = True
                    self.running = False
                    return False
                
                # 定期同步显示（降低频率避免性能问题）
                if self.mj_data.time - last_render_time > (1.0 / self.render_fps):
                    self.viewer.sync()
                    last_render_time = self.mj_data.time

            # 每秒报告一次进度
            if time.time() - last_report_time > 1.0:
                elapsed = time.time() - self.start_time
                sim_time_info = f", 仿真时间: {self.mj_data.time:.1f}s" if self.sync else ""
                print(f"   ⏱️  运行时间: {elapsed:.1f}s, 步数: {step_count}, 当前状态: {self.stm.state_idx+1}/{self.total_states}{sim_time_info}")
                last_report_time = time.time()

        # 报告结果
        elapsed_time = time.time() - self.start_time
        print(f"\\n📊 {self.robot_name.upper()}运行架构执行完成!")
        print(f"   总时间: {elapsed_time:.2f}s")
        print(f"   仿真时间: {self.mj_data.time:.2f}s")
        print(f"   总步数: {step_count}")
        print(f"   完成状态: {self.stm.state_idx}/{self.total_states}")
        print(f"   任务成功: {'✅ 是' if self.success else '❌ 否'}")
        if self.sync:
            time_ratio = self.mj_data.time / elapsed_time if elapsed_time > 0 else 0
            print(f"   时间比例: {time_ratio:.2f} (仿真时间/真实时间)")
        
        return self.success
    
    def reset(self):
        """重置环境和执行器状态"""
        # 重置到home位置
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key(0).id)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        # 重置状态机
        self.stm = SimpleStateMachine()
        self.stm.max_state_cnt = self.total_states
        
        # 重置控制状态
        self.target_control = np.zeros(self.mujoco_ctrl_dim)
        self.action = np.zeros(self.mujoco_ctrl_dim)
        self.joint_move_ratio = np.ones(self.mujoco_ctrl_dim)
        
        # 重置运行时状态
        self.running = True
        self.start_time = time.time()
        self.success = False
        self.viewer_closed = False  # 重置viewer关闭标志
        
        # 重置延时状态
        self.current_delay = 0.0
        self.delay_start_sim_time = None
        
        # 重新初始化动作
        self.action[:] = self.get_current_qpos()[:self.mujoco_ctrl_dim]
        
        # 重新初始化mocap target
        mink.move_mocap_to_frame(self.mj_model, self.mj_data, "target", "endpoint", "site")
        print("🔄 环境已重置，准备下一轮任务")

def generate_robot_task_model(robot_name, task_name):
    """生成指定机械臂的任务模型"""
    xml_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf/tmp", f"{robot_name}_{task_name}.xml")
    env = make_env(robot_name, task_name, xml_path)
    print(f"🏗️ 生成{robot_name.upper()}_{task_name.upper()}模型: {xml_path}")
    return xml_path

def setup_scene(model, data, task_name):
    """初始化场景"""
    # 重置到home位置
    mujoco.mj_resetDataKeyframe(model, data, model.key(0).id)
    mujoco.mj_forward(model, data)
    
    # 初始化mocap target
    mink.move_mocap_to_frame(model, data, "target", "endpoint", "site")
    
    print("🎬 场景初始化完成")

def create_simple_visualizer(mj_model, mj_data):
    """创建MuJoCo内置可视化器"""
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    
    # 检查是否有相机并设置默认视角
    if mj_model.ncam > 0:
        viewer.cam.fixedcamid = 0  # 使用id=0的相机
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        print(f"🎥 使用相机 id=0 作为默认视角 (共{mj_model.ncam}个相机)")
    else:
        print("📷 MJCF中未发现相机，使用自由视角")
    
    print("🎬 MuJoCo内置查看器创建成功")
    return viewer

def main(robot_name="airbot_play", task_name="place_block", sync=False, once=False, headless=False):
    """主函数 - 改进版本，支持模板化配置和CICD模式
    
    Args:
        robot_name: 机械臂名称
        task_name: 任务名称
        sync: 是否实时同步
        once: 是否单次执行
        headless: 是否无头模式（CICD用）
    """
    print(f"Welcome to discoverse {discoverse.__version__} !")
    print(discoverse.__logo__)

    print(f"🤖 启动{robot_name.upper()} {task_name}任务演示")
    print(f"📋 执行模式: {'单次执行' if once else '循环执行'}")
    print(f"📺 显示模式: {'无头模式' if headless else '可视化模式'}")
    print("=" * 70)
    
    # 验证配置文件
    config_path = os.path.join(
        discoverse.DISCOVERSE_ROOT_DIR, 
        f"discoverse/configs/tasks/{task_name}.yaml"
    )
    
    if os.path.exists(config_path):
        try:
            config = load_and_resolve_config(config_path)
            config = replace_variables(config)  # 🔧 添加变量替换
            if not validate_task_config(config, robot_name):
                print("❌ 配置验证失败")
                return
            print("✅ 配置验证通过")
        except Exception as e:
            print(f"⚠️ 配置加载失败: {e}")
    
    xml_path = generate_robot_task_model(robot_name, task_name)
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    print(f"✅ 模型加载成功! (nq={mj_model.nq}, nkey={mj_model.nkey})")

    # 初始化场景
    setup_scene(mj_model, mj_data, task_name)

    # 创建查看器（除非是无头模式）
    viewer = None if headless else create_simple_visualizer(mj_model, mj_data)
    if headless:
        print("🤖 无头模式运行")

    # 创建通用任务 - 使用预处理的配置
    try:
        from discoverse import DISCOVERSE_ROOT_DIR
        configs_root = os.path.join(DISCOVERSE_ROOT_DIR, "discoverse", "configs")
        robot_config_path = os.path.join(configs_root, "robots", f"{robot_name}.yaml")
        
        # 直接创建任务实例，传递预处理的配置
        task = UniversalTaskBase(
            robot_config_path=robot_config_path,
            task_config_path=None,  # 不从文件加载
            mj_model=mj_model,
            mj_data=mj_data
        )
        
        # 手动设置已处理的任务配置
        from discoverse.universal_manipulation.task_config import TaskConfigLoader
        task.task_config = TaskConfigLoader.from_dict(config)
        
        # 创建任务执行器
        task._create_executor()
        
        print(f"✅ 任务创建成功")
        
        # 设置viewer引用
        if viewer:
            task.robot_interface.set_viewer(viewer)
        
    except Exception as e:
        print(f"❌ 任务创建失败: {e}")
        if not headless:  # 只在非CICD模式打印详细错误
            traceback.print_exc()
        return

    # 创建通用运行时执行器
    try:
        executor = UniversalRuntimeTaskExecutor(task, viewer, mj_model, mj_data, robot_name, sync)

        # 任务循环执行
        task_count = 0
        if once:
            print(f"\\n🎯 开始单次任务执行")
        else:
            print(f"\\n🔁 开始循环任务执行模式")
            print(f"   提示: 关闭查看器窗口可退出程序")
        
        while True:
            task_count += 1
            print(f"\\n{'='*50}")
            print(f"🎯 第 {task_count} 轮任务开始")
            print(f"{'='*50}")
            
            # 运行任务
            success = executor.run()
            
            if success:
                print(f"\\n🎉 第 {task_count} 轮任务成功完成!")
                print(f"   任务目标已达成")
            else:
                print(f"\\n⚠️ 第 {task_count} 轮任务未完全成功")
            
            # 单次执行模式下直接退出
            if once:
                print(f"\\n📋 单次执行模式，任务完成后退出")
                break
            
            # 检查是否需要退出循环
            if executor.viewer_closed:
                print(f"\\n🛑 检测到查看器关闭，结束循环")
                break
            
            # 重置环境准备下一轮
            executor.reset()
        
        print(f"\\n📊 任务执行总结:")
        print(f"   总执行轮数: {task_count}")
        exit_reason = "单次执行完成" if once else "查看器关闭"
        print(f"   退出原因: {exit_reason}")
        
    except Exception as e:
        print(f"❌ 运行时执行失败: {e}")
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
    import argparse
    parser = argparse.ArgumentParser(description="通用机械臂任务演示 - 改进版本")
    parser.add_argument("-r", "--robot", type=str, default="airbot_play", 
                       choices=["airbot_play", "arx_x5", "arx_l5", "piper", "panda", "rm65", "xarm7", "iiwa14", "ur5e"],
                       help="选择机械臂类型")
    parser.add_argument("-t", "--task", type=str, default="place_block",
                       choices=["place_block", "cover_cup", "stack_block", "place_kiwi_fruit", "place_coffeecup", "close_laptop"],
                       help="选择任务类型")
    parser.add_argument("-s", "--sync", action="store_true", 
                       help="启用实时同步模式（仿真时间与真实时间一致）")
    parser.add_argument("-1", "--once", action="store_true",
                       help="单次执行模式（默认为循环执行）")
    parser.add_argument("--headless", action="store_true",
                       help="无头模式运行（CICD测试用）")
    args = parser.parse_args()

    main(args.robot, args.task, sync=args.sync, once=args.once, headless=args.headless)
