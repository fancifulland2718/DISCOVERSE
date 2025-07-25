import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import argparse
import multiprocessing as mp

from discoverse.robots import AirbotPlayIK
from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.robots_env.airbot_play_base import AirbotPlayCfg
from discoverse.utils import get_body_tmat, get_site_tmat, step_func, SimpleStateMachine
from discoverse.task_base import AirbotPlayTaskBase, recoder_airbot_play, copypy2
from discoverse.task_base.airbot_task_base import PyavImageEncoder
import traceback


class SimNode(AirbotPlayTaskBase):
    def domain_randomization(self):
        # random_bias_xy = np.zeros(2)  # 随机偏移量 创建一个包含两个元素的零向量
        random_bias_xy = np.zeros(2)
        random_bias_xy[0] = 2.*(np.random.random() - 0.5) * 0.00  # 减去 0.5 使得随机数在 -0.5 到 0.5 之间 乘以 2.0 使得随机数在 -1.0 到 1.0 之间
        random_bias_xy[1] = 2.*(np.random.random() - 0.5) * 0.00  # 减去 0.5 使得随机数在 -0.5 到 0.5 之间 乘以 2.0 使得随机数在 -1.0 到 1.0 之间
            # 0.08 0.05 0.04 0.03
        # 随机 红烧肉
        self.mj_data.qpos[self.nj+1+0] += random_bias_xy[0]
        self.mj_data.qpos[self.nj+1+1] += random_bias_xy[1]

        # 随机 碟子位置  碟子和红烧肉相同偏移量
        self.mj_data.qpos[self.nj+1*7+1+0] += random_bias_xy[0]
        self.mj_data.qpos[self.nj+1*7+1+1] += random_bias_xy[1]

        # 随机 碗位置    相对红烧肉偏离14个单位
        self.mj_data.qpos[self.nj+2*7+1+0] += 2.*(np.random.random() - 0.5) * 0.00  #X 方向：在 [-0.05, 0.05) 范围内随机浮动
        self.mj_data.qpos[self.nj+2*7+1+1] += 2.*(np.random.random() - 0.5) * 0.00  #Y 方向：在 [-0.05, 0.05) 范围内随机浮动

    def check_success(self):
        tmat_jujube = get_body_tmat(self.mj_data, "meat_1")
        tmat_bowl = get_body_tmat(self.mj_data, "plate_white")
        return (abs(tmat_bowl[2, 2]) > 0.99) and np.hypot(tmat_jujube[0, 3] - tmat_bowl[0, 3], tmat_jujube[1, 3] - tmat_bowl[1, 3]) < 0.02

cfg = AirbotPlayCfg()
cfg.use_gaussian_renderer = True
cfg.init_key = "ready"
cfg.gs_model_dict["background"]  = "scene/lab3/point_cloud.ply"
# cfg.gs_model_dict["drawer_1"]    = "hinge/drawer_1.ply"
# cfg.gs_model_dict["drawer_2"]    = "hinge/drawer_2.ply"
# cfg.gs_model_dict["meat"]        = "object/meat.ply"
cfg.gs_model_dict["plate_white"] = "object/plate_white.ply"
cfg.gs_model_dict["flower_bowl"] = "object/flower_bowl.ply"

cfg.mjcf_file_path = "mjcf/tasks_airbot_play/meat_place.xml"
cfg.obj_list     = ["meat_4","meat_3", "meat_2", "meat_1", "plate_white"]
cfg.timestep     = 1/240
cfg.decimation   = 4
cfg.sync         = True
cfg.headless     = False
cfg.render_set   = {
    "fps"    : 20,
    "width"  : 640,
    "height" : 480
}
cfg.obs_rgb_cam_id = [0, 1]
cfg.save_mjb_and_task_config = True

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    parser.add_argument("--save_segment", action="store_true", help="save segment videos")
    parser.add_argument('--use_gs', action='store_true', help='Use gaussian splatting renderer')
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False
    cfg.use_gaussian_renderer = args.use_gs
    
    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/meat_place")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.save_segment:
        cfg.obs_depth_cam_id = list(set(cfg.obs_rgb_cam_id + ([] if cfg.obs_depth_cam_id is None else cfg.obs_depth_cam_id)))
        from discoverse.randomain.utils import SampleforDR
        samples = SampleforDR(objs=cfg.obj_list[2:], robot_parts=cfg.rb_link_list, cam_ids=cfg.obs_rgb_cam_id, save_dir=os.path.join(save_dir, "segment"), fps=cfg.render_set["fps"])

    sim_node = SimNode(cfg)
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        copypy2(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))

    arm_ik = AirbotPlayIK()     # 逆运动学求解器

    trmat = Rotation.from_euler("xyz", [0., 1.4, 0.], degrees=False).as_matrix() # 末端执行器的姿态矩阵
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base")) # 基座到世界坐标系的变换矩阵

    stm = SimpleStateMachine() # 状态机
    stm.max_state_cnt = 36 # 状态数
    max_time = 20.0 #s

    action = np.zeros(7)
    act_lst, obs_lst = [], []
    process_list = []

    move_speed = 0.75
    sim_node.reset()
    while sim_node.running:
        if sim_node.reset_sig:
            # 重置状态
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []
            save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
            os.makedirs(save_path, exist_ok=True)
            encoders = {cam_id: PyavImageEncoder(20, cfg.render_set["width"], cfg.render_set["height"], save_path, cam_id) for cam_id in cfg.obs_rgb_cam_id}
            if args.save_segment:
                samples.reset()
        # 矩阵结构：tmat_meat[:3,3] 是红烧肉的世界坐标（X,Y,Z）；tmat_meat[:3,:3] 是姿态旋转矩阵
        try:
            if stm.trigger():
                if stm.state_idx == 0: # 伸到红烧肉上方
                    tmat_meat = get_body_tmat(sim_node.mj_data, "meat_1")
                    tmat_meat[:3, 3] = tmat_meat[:3, 3] + 0.12 * tmat_meat[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_meat
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    sim_node.target_control[6] = 0.5 # 1表示张开，0.5表示半闭合，0表示闭合
                elif stm.state_idx == 1: # 伸到红烧肉
                    tmat_meat = get_body_tmat(sim_node.mj_data, "meat_1")
                    tmat_meat[:3, 3] = tmat_meat[:3, 3] + 0.040 * tmat_meat[:3, 2] # 0.045 夹爪离红烧肉的高度偏移量
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_meat # 
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6]) #arm_ik.properIK 是机器人的逆运动学求解器，根据目标位置和姿态，计算出 6 个关节需要达到的角度。sim_node.target_control[:6] 存储这些目标角度，后续控制逻辑会驱动关节向这些角度移动。
                elif stm.state_idx == 2: # 抓住红烧肉
                    sim_node.target_control[6] = 0 # 1表示张开，0表示闭合 
                elif stm.state_idx == 3: # 抓稳红烧肉
                    sim_node.delay_cnt = int(0.35/sim_node.delta_t) #延迟计数器，让机器人保持抓取状态一段时间（0.35s）
                    #sim_node.delta_t ->是模拟的时间步长   计算出需要延迟的步数
                elif stm.state_idx == 4: # 提起来红烧肉
                    tmat_tgt_local[2,3] += 0.15 # z轴向上抬起0.15m  ， tmat_tgt_local 是机器人基座坐标系下的目标变换矩阵，[2,3] 对应 Z 轴（高度）坐标。
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 5: # 把红烧肉放到碗上空
                    tmat_bowl = get_body_tmat(sim_node.mj_data, "plate_white")
                    tmat_bowl[:3,3] = tmat_bowl[:3, 3] + np.array([0.0, 0.0, 0.13]) # tmat_bowl[:3,3] 是碗在世界坐标系下的 X、Y、Z 坐标，加 [0,0,0.13] 后得到碗上方 0.13 米的目标位置
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bowl # 确保目标位置适配机器人基座坐标系，便于逆运动学求解。
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 6: # 降低高度 把红烧肉放到碗上
                    tmat_tgt_local[2,3] -= 0.04
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 7: # 松开红烧肉
                    sim_node.target_control[6] = 0.5
                elif stm.state_idx == 8: # 抬升高度
                    tmat_tgt_local[2,3] += 0.05
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 9: #伸到meat_2上方
                    tmat_meat = get_body_tmat(sim_node.mj_data, "meat_2")
                    tmat_meat[:3, 3] = tmat_meat[:3, 3] + 0.12 * tmat_meat[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_meat
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 10: #伸到meat_2
                    tmat_meat = get_body_tmat(sim_node.mj_data, "meat_2")
                    tmat_meat[:3, 3] = tmat_meat[:3, 3] + 0.040 * tmat_meat[:3, 2] # 0.045 夹爪离红烧肉的高度偏移量
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_meat
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 11: # 抓住meat_2
                    sim_node.target_control[6] = 0 # 1表示张开，0表示闭合
                elif stm.state_idx == 12: # 抓稳meat_2
                    sim_node.delay_cnt = int(0.35/sim_node.delta_t) #延迟计数器，让机器人保持抓取状态一段时间（0.35s）
                elif stm.state_idx == 13: # 提起来meat_2
                    tmat_tgt_local[2,3] += 0.15 # z轴向上抬起0.15m
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 14: # 把meat_2放到碗上空
                    tmat_bowl = get_body_tmat(sim_node.mj_data, "plate_white")
                    tmat_bowl[:3,3] = tmat_bowl[:3, 3] + np.array([0.0, 0.0, 0.13])    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bowl # 确保目标位置适配机器人基座坐标系，便于逆运动学求解。
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 15: # 降低高度 把meat_2放到碗上
                    tmat_tgt_local[2,3] -= 0.04
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 16: # 松开meat_2
                    sim_node.target_control[6] = 0.5
                elif stm.state_idx == 17: # 抬升高度
                    tmat_tgt_local[2,3] += 0.05
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 18: #伸到meat_3上方
                    tmat_meat = get_body_tmat(sim_node.mj_data, "meat_3")
                    tmat_meat[:3, 3] = tmat_meat[:3, 3] + 0.12 * tmat_meat[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_meat
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 19: #伸到meat_3
                    tmat_meat = get_body_tmat(sim_node.mj_data, "meat_3")
                    tmat_meat[:3, 3] = tmat_meat[:3, 3] + 0.040 * tmat_meat[:3, 2] # 0.045 夹爪离红烧肉的高度偏移量
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_meat
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 20: # 抓住meat_3
                    sim_node.target_control[6] = 0 # 1表示张开，0表示闭合
                elif stm.state_idx == 21: # 抓稳meat_3
                    sim_node.delay_cnt = int(0.35/sim_node.delta_t) #延迟计数器，让机器人保持抓取状态一段时间（0.35s）
                elif stm.state_idx == 22: # 提起来meat_3
                    tmat_tgt_local[2,3] += 0.15 # z轴向上抬起0.15m
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 23: # 把meat_3放到碗上空
                    tmat_bowl = get_body_tmat(sim_node.mj_data, "plate_white")
                    tmat_bowl[:3,3] = tmat_bowl[:3, 3] + np.array([0.0, 0.0, 0.13])    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bowl # 确保目标位置适配机器人基座坐标系，便于逆运动学求解。
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 24: # 降低高度 把meat_3放到碗上
                    tmat_tgt_local[2,3] -= 0.04
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 25: # 松开meat_3
                    sim_node.target_control[6] = 0.5
                elif stm.state_idx == 26: # 抬升高度
                    tmat_tgt_local[2,3] += 0.05
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 27: #伸到meat_4上方
                    tmat_meat = get_body_tmat(sim_node.mj_data, "meat_4")
                    tmat_meat[:3, 3] = tmat_meat[:3, 3] + 0.12 * tmat_meat[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_meat
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 28: #伸到meat_4
                    tmat_meat = get_body_tmat(sim_node.mj_data, "meat_4")
                    tmat_meat[:3, 3] = tmat_meat[:3, 3] + 0.040 * tmat_meat[:3, 2] # 0.045 夹爪离红烧肉的高度偏移量
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_meat
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 29: # 抓住meat_4
                    sim_node.target_control[6] = 0 # 1表示张开，0表示闭合
                elif stm.state_idx == 30: # 抓稳meat_4
                    sim_node.delay_cnt = int(0.35/sim_node.delta_t) #延迟计数器，让机器人保持抓取状态一段时间（0.35s）
                elif stm.state_idx == 31: # 提起来meat_4
                    tmat_tgt_local[2,3] += 0.15 # z轴向上抬起0.15m
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 32: # 把meat_4放到碗上空
                    tmat_bowl = get_body_tmat(sim_node.mj_data, "plate_white")
                    tmat_bowl[:3,3] = tmat_bowl[:3, 3] + np.array([0.0, 0.0, 0.13])    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bowl # 确保目标位置适配机器人基座坐标系，便于逆运动学求解。
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 33: # 降低高度 把meat_4放到碗上
                    tmat_tgt_local[2,3] -= 0.04
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 34: # 松开meat_4
                    sim_node.target_control[6] = 0.5
                elif stm.state_idx == 35: # 抬升高度
                    tmat_tgt_local[2,3] += 0.05
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    
                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)

            elif sim_node.mj_data.time > max_time:  #检测是否超时
                raise ValueError("Time out")

            else:
                stm.update()

            if sim_node.checkActionDone():
                stm.next()

        except ValueError as ve:
            traceback.print_exc()
            sim_node.reset()

        for i in range(sim_node.nj-1):
            action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)
        action[6] = sim_node.target_control[6]

        obs, _, _, _, _ = sim_node.step(action)

        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            imgs = obs.pop('img')
            for cam_id, img in imgs.items():
                encoders[cam_id].encode(img, obs["time"])
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)
            if args.save_segment:
                samples.sampling(sim_node)

        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                recoder_airbot_play(save_path, act_lst, obs_lst, cfg)
                for encoder in encoders.values():
                    encoder.close()
                if args.save_segment:
                    seg_process = mp.Process(target=samples.save)
                    seg_process.start()
                    process_list.append(seg_process)

                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
            else:
                print(f"{data_idx} Failed")

            sim_node.reset()

    for p in process_list:
        p.join()
