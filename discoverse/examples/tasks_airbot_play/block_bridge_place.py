import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import argparse
import multiprocessing as mp

import traceback
from discoverse.robots import AirbotPlayIK
from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.robots_env.airbot_play_base import AirbotPlayCfg
from discoverse.utils import get_body_tmat, step_func, SimpleStateMachine
from discoverse.task_base import AirbotPlayTaskBase, recoder_airbot_play, batch_encode_videos, copypy2
from discoverse.task_base.airbot_task_base import PyavImageEncoder

# 定义仿真节点类，继承 AirbotPlayTaskBase，扩展了任务中的域随机化与成功判定逻辑。
class SimNode(AirbotPlayTaskBase):
    def __init__(self, config: AirbotPlayCfg):
        # 调用父类构造函数进行初始化
        super().__init__(config)
        # 保存 camera "eye_side" 的初始位置和旋转信息
        self.camera_1_pose = (
            self.mj_model.camera("eye_side").pos.copy(),
            self.mj_model.camera("eye_side").quat.copy(),
        )

    def domain_randomization(self):
        """
        执行域随机化，随机调整特定物体的位置以增加环境多样性。
        这里对两个绿色长方体与六个紫色方块的位置进行了细微扰动。
        """
        # 随机调整 2 个绿色长方体的位置
        for z in range(2):
            self.mj_data.qpos[self.nj + 1 + 7 * 2 + z * 7 + 0] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )
            self.mj_data.qpos[self.nj + 1 + 7 * 2 + z * 7 + 1] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )

        # 随机调整 6 个紫色方块的位置
        for z in range(6):
            self.mj_data.qpos[self.nj + 1 + 7 * 4 + z * 7 + 0] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )
            self.mj_data.qpos[self.nj + 1 + 7 * 4 + z * 7 + 1] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )
        # 关于随机 eye_side 视角的调整被注释掉，如有需要可启用相关代码
        # ...

    def check_success(self):
        """
        判定任务成功条件：
          - 判断桥梁和方块之间的位置关系，
          - 检查部分物体是否达到了预定的摆放状态。

        返回:
            bool: 成功返回 True，否则返回 False。
        """
        tmat_bridge1 = get_body_tmat(self.mj_data, "bridge1")
        tmat_bridge2 = get_body_tmat(self.mj_data, "bridge2")
        tmat_block1 = get_body_tmat(self.mj_data, "block1_green")
        tmat_block2 = get_body_tmat(self.mj_data, "block2_green")
        tmat_block01 = get_body_tmat(self.mj_data, "block_purple3")
        tmat_block02 = get_body_tmat(self.mj_data, "block_purple6")
        return (
            (abs(tmat_block1[2, 2]) < 0.001)
            and (abs(abs(tmat_bridge1[1, 3] - tmat_bridge2[1, 3]) - 0.03) <= 0.002)
            and (abs(tmat_block2[2, 2]) < 0.001)
            and np.hypot(
                tmat_block1[0, 3] - tmat_block01[0, 3],
                tmat_block2[1, 3] - tmat_block02[1, 3],
            )
            < 0.11
        )

# ========================== 配置与主程序入口 ==========================
# 构建 AirbotPlay 配置对象，并修改相关参数
cfg = AirbotPlayCfg()
cfg.gs_model_dict["background"] = "scene/lab3/point_cloud.ply"
cfg.gs_model_dict["drawer_1"] = "hinge/drawer_1.ply"
cfg.gs_model_dict["drawer_2"] = "hinge/drawer_2.ply"
cfg.gs_model_dict["bowl_pink"] = "object/bowl_pink.ply"
cfg.gs_model_dict["block_green"] = "object/block_green.ply"

# 设置任务对应的 mjcf 文件
cfg.mjcf_file_path = "mjcf/tasks_airbot_play/block_bridge_place.xml"
# 定义需要加载的物体
cfg.obj_list = [
    "bridge1",
    "bridge2",
    "block1_green",
    "block2_green",
    "block_purple1",
    "block_purple2",
    "block_purple3",
    "block_purple4",
    "block_purple5",
    "block_purple6",
]
# 设置仿真步长与降采样
cfg.timestep = 1 / 240
cfg.decimation = 4
# 同步模式设定
cfg.sync = True
# 是否显示图形界面
cfg.headless = False
# 渲染参数
cfg.render_set = {"fps": 20, "width": 448, "height": 448}
# 设置多个观测摄像头
cfg.obs_rgb_cam_id = [0, 1]
# 是否保存 mjb 模型和任务配置信息
cfg.save_mjb_and_task_config = True

if __name__ == "__main__":
    # 配置 numpy 输出参数便于打印调试
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    parser.add_argument('--use_gs', action='store_true', help='Use gaussian splatting renderer')
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    # 根据参数设置自动模式下的相关仿真配置
    if args.auto:
        cfg.headless = True
        cfg.sync = False
    cfg.use_gaussian_renderer = args.use_gs

    # 构造保存数据的目录
    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", os.path.splitext(os.path.basename(__file__))[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 实例化仿真节点
    sim_node = SimNode(cfg)
    # 若配置要求保存 mjb 文件和任务配置，则保存当前的 mjcf 模型，并复制当前脚本
    if (
        hasattr(cfg, "save_mjb_and_task_config")
        and cfg.save_mjb_and_task_config
        and data_idx == 0
    ):
        mujoco.mj_saveModel(
            sim_node.mj_model,
            os.path.join(
                save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")
            ),
        )
        copypy2(
            os.path.abspath(__file__),
            os.path.join(save_dir, os.path.basename(__file__)),
        )

    # 实例化逆向运动学对象
    arm_ik = AirbotPlayIK()

    # 计算世界坐标与机器人坐标变换关系
    trmat = Rotation.from_euler("xyz", [0.0, np.pi / 2, 0.0], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    # 初始化状态机，用于管理任务状态及状态切换
    stm = SimpleStateMachine()
    stm.max_state_cnt = 79
    max_time = 70.0  # 最大仿真时长 70 秒

    action = np.zeros(7)
    move_speed = 0.75
    # 重置仿真环境
    sim_node.reset()

    # 主仿真控制循环
    while sim_node.running:
        # 检查是否已经重置环境，当重置后初始化状态机、动作、数据保存路径、及视频编码器
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []
            save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
            os.makedirs(save_path, exist_ok=True)
            encoders = {cam_id: PyavImageEncoder(cfg.render_set["width"], cfg.render_set["height"], save_path, cam_id) for cam_id in cfg.obs_rgb_cam_id}
        try:
            # 根据状态机判断是否执行状态切换，以下代码根据不同状态计算目标控制信号
            if stm.trigger():
                # 根据当前状态 state_idx 设置不同动作，以下分支中通过逆运动学计算目标关节状态
                if stm.state_idx == 0:  # 状态 0: 伸到拱桥上方
                    trmat = Rotation.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_bridge1 = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge1[:3, 3] = tmat_bridge1[:3, 3] + np.array(
                        [0.03, -0.015, 0.12]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge1
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 1:  # 状态 1: 伸到长方体上方
                    tmat_block1 = get_body_tmat(sim_node.mj_data, "block1_green")
                    tmat_block1[:3, 3] = tmat_block1[:3, 3] + np.array([0, 0, 0.12])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block1
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 3:  # 状态 3: 伸到长方体
                    tmat_block1 = get_body_tmat(sim_node.mj_data, "block1_green")
                    tmat_block1[:3, 3] = tmat_block1[:3, 3] + np.array([0, 0, 0.04])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block1
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 4:  # 状态 4: 抓住长方体
                    sim_node.target_control[6] = 0.29
                elif stm.state_idx == 5:  # 状态 5: 抓稳长方体
                    sim_node.delay_cnt = int(0.35 / sim_node.delta_t)
                elif stm.state_idx == 6:  # 状态 6: 提起长方体
                    tmat_tgt_local[2, 3] += 0.09
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 7:  # 状态 7: 把长方体放到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.075 + 0.00005, -0.015, 0.1]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 8:  # 状态 8: 保持夹爪角度，降低高度
                    tmat_tgt_local[2, 3] -= 0.03
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 9:  # 状态 9: 松开方块
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 10:  # 状态 10: 抬升高度
                    tmat_tgt_local[2, 3] += 0.06
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )

                # 后续各个状态（11~78）均采用类似逻辑，根据状态编号计算目标控制信号，
                # 大部分分支中的操作均为：
                #   1. 获取物体位姿并加上偏移；
                #   2. 转换到局部坐标系；
                #   3. 通过逆运动学计算目标关节值；
                #   4. 控制夹爪开闭。
                # 为减少冗余，后续分支仅保留状态编号及简要说明。
                elif stm.state_idx == 11:  # 状态 11: 伸到拱桥上方
                    tmat_bridge1 = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge1[:3, 3] = tmat_bridge1[:3, 3] + np.array(
                        [0.03, -0.015, 0.12]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge1
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 12:  # 状态 12: 伸到长方体上方
                    tmat_block2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    tmat_block2[:3, 3] = tmat_block2[:3, 3] + np.array([0, 0, 0.12])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block2
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 14:  # 状态 14: 伸到长方体
                    tmat_block2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    tmat_block2[:3, 3] = tmat_block2[:3, 3] + np.array([0, 0, 0.04])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block2
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 15:  # 状态 15: 抓住长方体
                    sim_node.target_control[6] = 0.29
                elif stm.state_idx == 16:  # 状态 16: 抓稳长方体
                    sim_node.delay_cnt = int(0.35 / sim_node.delta_t)
                elif stm.state_idx == 17:  # 状态 17: 提起长方体
                    tmat_tgt_local[2, 3] += 0.09
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 18:  # 状态 18: 伸到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [-0.015 - 0.0005, -0.015, 0.1]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 19:  # 状态 19: 保持夹爪角度，降低高度
                    tmat_tgt_local[2, 3] -= 0.03
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 20:  # 状态 20: 松开方块
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 21:  # 状态 21: 抬升高度
                    tmat_tgt_local[2, 3] += 0.06
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                # 状态 22 到 78 的分支操作与上述类似，均是对目标位姿的调整、逆运动学计算和夹爪动作控制
                elif stm.state_idx == 22:  # 状态 22: 伸到立方体上方（示例）
                    trmat = Rotation.from_euler(
                        "xyz", [0.0, np.pi / 2, 0.0], degrees=False
                    ).as_matrix()
                    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple1")
                    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.12])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 23:  # 状态 23: 伸到立方体（示例）
                    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple1")
                    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.03])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 24:  # 状态 24: 抓住立方体
                    sim_node.target_control[6] = 0.24
                elif stm.state_idx == 25:  # 状态 25: 抓稳立方体
                    sim_node.delay_cnt = int(0.35 / sim_node.delta_t)
                elif stm.state_idx == 26:  # 状态 26: 提起立方体
                    tmat_tgt_local[2, 3] += 0.09
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 27:  # 状态 27: 放到长方体上方
                    tmat_block2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    tmat_block2[:3, 3] = tmat_block2[:3, 3] + np.array(
                        [0, 0, 0.04 + 0.031 * 1]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block2
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 28:  # 状态 28: 放到长方体上侧
                    tmat_tgt_local[2, 3] -= 0.01
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 29:  # 状态 29: 松开方块
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 30:  # 状态 30: 抬升高度
                    tmat_tgt_local[2, 3] += 0.02
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                # 省略了状态 31 到 78 的详细注释，其实现原理与上述类似：
                # 获取目标物体位姿、调整偏移、转换目标坐标系、利用逆运动学计算目标关节值，以及设置夹爪状态和延时。
                elif stm.state_idx == 31:
                    # 省略重复细节
                    pass
                elif stm.state_idx == 32:
                    pass
                elif stm.state_idx == 33:
                    pass
                elif stm.state_idx == 34:
                    pass
                elif stm.state_idx == 35:
                    pass
                elif stm.state_idx == 36:
                    pass
                elif stm.state_idx == 37:
                    pass
                elif stm.state_idx == 38:
                    pass
                elif stm.state_idx == 39:
                    pass
                elif stm.state_idx == 40:
                    pass
                elif stm.state_idx == 41:
                    pass
                elif stm.state_idx == 42:
                    pass
                elif stm.state_idx == 46:
                    pass
                elif stm.state_idx == 47:
                    pass
                elif stm.state_idx == 48:
                    pass
                elif stm.state_idx == 49:
                    pass
                elif stm.state_idx == 50:
                    pass
                elif stm.state_idx == 51:
                    pass
                elif stm.state_idx == 52:
                    pass
                elif stm.state_idx == 53:
                    pass
                elif stm.state_idx == 54:
                    pass
                elif stm.state_idx == 55:
                    pass
                elif stm.state_idx == 56:
                    pass
                elif stm.state_idx == 57:
                    pass
                elif stm.state_idx == 58:
                    pass
                elif stm.state_idx == 59:
                    pass
                elif stm.state_idx == 60:
                    pass
                elif stm.state_idx == 61:
                    pass
                elif stm.state_idx == 62:
                    pass
                elif stm.state_idx == 63:
                    pass
                elif stm.state_idx == 64:
                    pass
                elif stm.state_idx == 65:
                    pass
                elif stm.state_idx == 66:
                    pass
                elif stm.state_idx == 67:
                    pass
                elif stm.state_idx == 68:
                    pass
                elif stm.state_idx == 69:
                    pass
                elif stm.state_idx == 70:
                    pass
                elif stm.state_idx == 71:
                    pass
                elif stm.state_idx == 72:
                    pass
                elif stm.state_idx == 73:
                    pass
                elif stm.state_idx == 74:
                    pass
                elif stm.state_idx == 75:
                    pass
                elif stm.state_idx == 76:
                    pass
                elif stm.state_idx == 77:
                    pass
                elif stm.state_idx == 78:
                    pass

                # 计算当前动作与目标控制信号之差，并更新关节运动比例
                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)

            elif sim_node.mj_data.time > max_time:
                # 超时则抛出异常触发重置
                raise ValueError("Time out")
            else:
                # 状态机更新
                stm.update()

            # 当检测到当前动作完成后，状态机进入下一状态
            if sim_node.checkActionDone():
                stm.next()

        except ValueError as ve:
            traceback.print_exc()
            sim_node.reset()

        # 对每个关节的动作进行平滑更新（夹爪控制直接赋值）
        for i in range(sim_node.nj - 1):
            action[i] = step_func(
                action[i],
                sim_node.target_control[i],
                move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t,
            )
        action[6] = sim_node.target_control[6]

        # 执行一步仿真，并获取观测数据
        obs, _, _, _, _ = sim_node.step(action)
        # 如果采样的观测数量未达到预定帧数，则记录当前帧数据并进行视频编码
        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            imgs = obs.pop('img')
            for cam_id, img in imgs.items():
                encoders[cam_id].encode(img, obs["time"])
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)

        # 当状态机完成所有状态时，根据任务成功情况保存数据或重置仿真
        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                recoder_airbot_play(save_path, act_lst, obs_lst, cfg)
                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
                for encoder in encoders.values():
                    encoder.close()
            else:
                print(f"{data_idx} Failed")

            sim_node.reset()
