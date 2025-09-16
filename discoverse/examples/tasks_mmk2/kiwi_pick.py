import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import argparse
import multiprocessing as mp
import traceback

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.robots_env.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase, recoder_mmk2, copypy2
from discoverse.utils import get_body_tmat, step_func, SimpleStateMachine


# 定义仿真节点类，继承自MMK2TaskBase
class SimNode(MMK2TaskBase):

    # 域随机化方法，用于随机化物体初始位置
    def domain_randomization(self):
        global kiwi_x_bios, kiwi_y_bios, kiwi_a_bios
        # 随机木盘位置
        plate_white_x_bios = (np.random.random()) * 0.02  # x方向偏移
        plate_white_y_bios = (np.random.random() - 1) * 0.05  # y方向偏移
        self.mj_data.qpos[self.njq+7*0+0] += plate_white_x_bios
        self.mj_data.qpos[self.njq+7*0+1] += plate_white_y_bios
        self.mj_data.qpos[self.njq+7*0+2] += 0.01  # z方向微调

        # 随机猕猴桃位置和朝向
        kiwi_x_bios = np.random.uniform(-0.2, -0.05)
        kiwi_y_bios = np.random.uniform(-0.08, 0.12)
        kiwi_a_bios = np.random.choice([0, 1])

        self.mj_data.qpos[self.njq+7*1+0] += kiwi_x_bios
        self.mj_data.qpos[self.njq+7*1+1] += kiwi_y_bios
        self.mj_data.qpos[self.njq+7*1+6] += kiwi_a_bios

    # 检查任务是否成功（猕猴桃是否放到木盘上)
    def check_success(self):
        tmat_kiwi = get_body_tmat(self.mj_data, "kiwi")
        tmat_plate_white = get_body_tmat(self.mj_data, "plate_white")
        # 计算猕猴桃和木盘的水平距离
        distance= np.hypot(tmat_kiwi[0, 3] - tmat_plate_white[0, 3], tmat_kiwi[1, 3] - tmat_plate_white[1, 3])
        # print(distance)
        return distance < 0.018  # 距离小于阈值则判定成功


# 配置MMK2机器人及仿真参数
cfg = MMK2Cfg()
cfg.use_gaussian_renderer = False  # 是否使用高斯渲染器
cfg.gs_model_dict["plate_white"]   = "object/plate_white.ply"  # 木盘模型路径
cfg.gs_model_dict["kiwi"]          = "object/kiwi.ply"         # 猕猴桃模型路径
cfg.gs_model_dict["background"]    = "scene/Lab3/environment.ply"  # 背景模型路径

cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_kiwi.xml"  # mujoco模型文件
cfg.obj_list    = ["plate_white", "kiwi"]  # 参与物体列表
cfg.sync     = True  # 是否同步仿真（加速)
cfg.headless = False  # 是否无头模式
cfg.render_set  = {
    "fps"    : 25,    # 渲染帧率
    "width"  : 640,   # 渲染宽度
    "height" : 480    # 渲染高度
}
cfg.obs_rgb_cam_id = [0,1,2]  # 观测相机ID
cfg.save_mjb_and_task_config = True  # 是否保存mjb和任务配置

# 机器人初始状态配置
cfg.init_state["base_position"] = [0.7, -0.5, 0.0]
cfg.init_state["base_orientation"] = [0.707, 0.0, 0.0, -0.707]
cfg.init_state["lft_arm_qpos"] = [0.0, -0.166, 0.032, 0.0, 1.571, 2.223]
cfg.init_state["rgt_arm_qpos"] = [0.0, -0.166, 0.032, 0.0, -1.571, -2.223]

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")  # 数据索引
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")  # 数据集大小
    parser.add_argument("--auto", action="store_true", help="auto run")  # 自动运行
    parser.add_argument('--use_gs', action='store_true', help='Use gaussian splatting renderer')  # 是否使用高斯渲染
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True  # 自动运行时无头
        cfg.sync = False     # 自动运行时不同步
    cfg.use_gaussian_renderer = args.use_gs

    # 数据保存目录
    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/pick_kiwi")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建仿真节点
    sim_node = SimNode(cfg)
    # 保存mjb模型和任务配置
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        copypy2(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))

    # 状态机初始化
    stm = SimpleStateMachine()
    stm.max_state_cnt = 9  # 状态数
    max_time = 20.0 # 最大仿真时间（秒）

    action = np.zeros_like(sim_node.target_control)  # 控制指令初始化
    process_list = []  # 进程列表

    pick_lip_arm = "l"  # 选择左臂（未使用）
    move_speed = 1.     # 运动速度
    obs = sim_node.reset()  # 环境重置
    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []  # 记录动作和观测

        try:
            if stm.trigger():
                # 状态机各状态对应的动作
                if stm.state_idx == 0:  # 降低头部高度
                    sim_node.tctr_head[1] = -0.8
                    sim_node.tctr_slide[0] = 0.2

                elif stm.state_idx == 1:  # 右臂移动到猕猴桃上方
                    tmat_kiwi = get_body_tmat(sim_node.mj_data, "kiwi")
                    if kiwi_a_bios == 1:
                        target_posi = tmat_kiwi[:3, 3] + np.array([0.0, 0.02, 0.11])  # y,x,z
                    else:
                        target_posi = tmat_kiwi[:3, 3] + np.array([0.02, 0.0, 0.1])  # y,x,z
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [0., -0.0551, 0.]).as_matrix())
                    sim_node.tctr_rgt_gripper[:] = 1  # 张开夹爪

                elif stm.state_idx == 2:  # 再次降低头部高度
                    sim_node.tctr_head[1] = -0.8
                    sim_node.tctr_slide[0] = 0.25

                elif stm.state_idx == 3:  # 抓取猕猴桃
                    if kiwi_a_bios == 1:
                        sim_node.tctr_rgt_gripper[:] = 0.65
                    else:
                        sim_node.tctr_rgt_gripper[:] = 0.4

                elif stm.state_idx == 4:  # 提起猕猴桃
                    sim_node.tctr_slide[0] = 0.1

                elif stm.state_idx == 5:  # 移动到木盘上方
                    tmat_plate_white = get_body_tmat(sim_node.mj_data, "plate_white")
                    target_posi = tmat_plate_white[:3, 3] + np.array([0.01, 0.01, 0.11])
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r",
                                             sim_node.sensor_rgt_arm_qpos,
                                             Rotation.from_euler('zyx', [0., -0.0551, 0.]).as_matrix())

                elif stm.state_idx == 6:  # 降低头部高度
                    sim_node.tctr_head[1] = -0.8
                    sim_node.tctr_slide[0] = 0.15

                elif stm.state_idx == 7:  # 放开猕猴桃
                    sim_node.tctr_rgt_gripper[:] = 1.0
                    sim_node.delay_cnt = int(0.2 / sim_node.delta_t)  # 延迟一段时间

                elif stm.state_idx == 8:  # 升高头部
                    sim_node.tctr_head[1] = -0.8
                    sim_node.tctr_slide[0] = 0.1


                # 计算关节运动比例
                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)
                sim_node.joint_move_ratio[2] *= 0.25  # 降低某关节速度

            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time out")  # 超时异常

            else:
                stm.update()  # 状态机更新

            if sim_node.checkActionDone():
                stm.next()  # 动作完成，进入下一个状态

        except ValueError as ve:
            # traceback.print_exc()
            sim_node.reset()

        for i in range(2, sim_node.njctrl):
            action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)
        yaw = Rotation.from_quat(np.array(obs["base_orientation"])[[1,2,3,0]]).as_euler("xyz")[2] + np.pi / 2
        action[1] = -10 * yaw

        obs, _, _, _, _ = sim_node.step(action)
        
        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)

        # 判断是否完成一轮任务
        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                # 成功则保存数据
                save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
                process = mp.Process(target=recoder_mmk2, args=(save_path, act_lst, obs_lst, cfg))
                process.start()
                process_list.append(process)

                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
            else:
                print(f"{data_idx} Failed")

            obs = sim_node.reset()  # 重置环境

    # 等待所有数据保存进程结束
    for p in process_list:
        p.join()
