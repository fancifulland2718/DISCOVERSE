import os
import sys
import time
import json
import platform
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
import mujoco.viewer
np.set_printoptions(precision=5, suppress=True, linewidth=500)

import mink
import discoverse
from discoverse.envs import make_env
from discoverse.examples.mocap_ik.mocap_ik_utils import \
    add_mocup_body_to_mjcf, \
    generate_mocap_xml
from discoverse.examples.force_control.impedance_control import ImpedanceController
from discoverse.utils import (
    get_mocap_tmat, 
    get_site_tmat, 
    get_body_tmat,
    get_control_idx,
    get_sensor_idx
)
from discoverse.universal_manipulation.recorder import PyavImageEncoder
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSETS_DIR

def recoder_state_file(save_path, obs_lst):
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
        save_dict = {
            "time" : [],
            "obs"  : {
                "jq" : [],      # 6个关节角度
                "jv" : [],      # 6个关节速度
                "tau": [],      # 6个关节力矩
                "eef_pos" : [], # 末端执行器位置xyz  armbase坐标系
                "eef_quat": [], # 末端执行器姿态wxyz armbase坐标系
                "eef_vel" : [], # 末端执行器速度xyz  armbase坐标系
                "eef_gyro": [], # 末端imu角速度
                "eef_acc" : [], # 末端imu加速度
                "end_force" : [], # 动力学算出来的6维度wrench 训练只用前三维
            },
            "act"  : [] # 末端位姿控制 xyz wxyz armbase坐标系
        }
        for obs in obs_lst:
            save_dict["time"].append(obs['time'])
            save_dict["obs"]["jq"].append(obs['jq'])
            save_dict["obs"]["jv"].append(obs['jv'])
            save_dict["obs"]["tau"].append(obs['tau'])
            save_dict["obs"]["eef_pos"].append(obs['eef_pos'])
            save_dict["obs"]["eef_quat"].append(obs['eef_quat'])
            save_dict["obs"]["eef_vel"].append(obs['eef_vel'])
            save_dict["obs"]["eef_gyro"].append(obs['eef_gyro'])
            save_dict["obs"]["eef_acc"].append(obs['eef_acc'])
            save_dict["obs"]["end_force"].append(obs['end_force'])
            save_dict["act"].append(obs['action'])
        json.dump(save_dict, fp)


SOLVER = "quadprog"
POS_THRESHOLD = 1e-3
ORI_THRESHOLD = 1e-3
MAX_ITERS = 10
def converge_ik(configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters):
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3)
        configuration.integrate_inplace(vel, dt)
        err = tasks[0].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

        if pos_achieved and ori_achieved:
            return True
    return False

if __name__ == "__main__":
    """
    机械臂的仿真主程序
    
    该程序创建一个机械臂模型的MuJoCo仿真环境，添加运动捕捉(mocap)目标，
    并使用逆运动学(IK)控制机器人的单臂跟踪目标位置和姿态。
    """
    print(f"Welcome to discoverse {discoverse.__version__} !")
    print(discoverse.__logo__)

    parser = argparse.ArgumentParser(
        description="Airbot Play 机器人MuJoCo仿真主程序\n"
                    "用法示例：\n"
                    "  python mocap_ik_airbot_play.py [--mjcf]\n"
                    "参数说明：\n"
                    "  mjcf  (可选) 指定MJCF模型文件路径，若不指定则使用默认模型。"
    )
    parser.add_argument(
        "-r", "--robot",
        type=str,
        default=None,
        help="输入机器人模型名称",
        choices=["airbot_play", "airbot_play_force", "arx_l5", "arx_x5", "iiwa14", "panda", "piper", "rm65", "ur5e", "xarm7"],
    )
    parser.add_argument(
        "-t", "--task",
        type=str,
        default=None,
        help="输入目标名称",
        choices=["block_bridge_place", "close_laptop", "cover_cup", "open_drawer", "peg_in_hole", "pick_jujube", "place_block", "place_coffeecup", "place_jujube", "place_jujube_coffeecup", "place_kiwi_fruit", "push_mouse", "stack_block"],
    )
    parser.add_argument(
        "-m", "--mjcf",
        type=str,
        default=None,
        help="输入MJCF文件的路径（可选）。如未指定，则使用默认的robot_airbot_play.xml"
    )
    parser.add_argument(
        "-y",
        action="store_true",
        help="在macOS上跳过mjpython提示，直接尝试启动查看器",
    )
    parser.add_argument(
        "--mouse-3d",
        action="store_true",
        help="启用3D鼠标进行机械臂控制（需要3D鼠标硬件支持）"
    )
    parser.add_argument(
        "--hide-mocap",
        action="store_true",
        help="隐藏运动捕捉目标"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="启用记录功能"
    )
    parser.add_argument(
        "--record-frequency",
        type=int, default=24,
        help="设置记录频率（单位：Hz）"
    )
    parser.add_argument(
        "--camera-names",
        type=str, nargs='*', default=[],
        help="指定需要渲染的相机名称列表（可选）"
    )

    args = parser.parse_args()
    enable_record = args.record or len(args.camera_names) > 0

    # 检查是否在macOS上运行并给出适当的提示
    if platform.system() == "Darwin" and not args.y:
        print("\n===================================================")
        print("注意: 在macOS上运行MuJoCo查看器需要使用mjpython")
        print("请使用以下命令运行此脚本:")
        print(f"mjpython {' '.join(sys.argv)}")
        print("===================================================\n")
        
        user_input = input("是否继续尝试启动查看器? (y/n): ")
        if user_input.lower() != 'y':
            print("退出程序。")
            sys.exit(0)

    if args.mouse_3d:
        import pyspacemouse
        success = pyspacemouse.open()
        if not success:
            print("3D鼠标打开失败，请检查设备连接")
            sys.exit(1)

    # 打印MuJoCo查看器的使用提示信息
    print("\n===================== MuJoCo 查看器使用说明 =====================")
    print("1. 双击选中机械臂末端的绿色方块，按下ctrl和鼠标左键，拖动鼠标可以旋转机械臂")
    print("   按下ctrl和鼠标右键，拖动鼠标可以平移机械臂末端")
    print("2. 按 Tab 键切换左侧 UI 的可视化界面。")
    print("   按 Shift+Tab 键切换右侧 UI 的可视化界面。")
    print("3. 在左侧 UI 中点击 'Copy State' 可以当前的机器人状态复制到剪贴板。")
    print("4. 在右侧 UI 的 control 面板中可以调节 gripper 滑块控制夹爪的开合。")
    print("================================================================\n")

    robot_name = args.robot
    task_name = args.task
    if robot_name is not None and task_name is not None:
        mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "tmp", f"{robot_name}_{task_name}.xml")
        env = make_env(robot_name, task_name, mjcf_path)
        env.export_xml(mjcf_path)
    elif robot_name is not None:
        mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "manipulator", f"robot_{robot_name}.xml")
    elif args.mjcf is not None:
        mjcf_path = args.mjcf
    else:
        mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "manipulator", "robot_airbot_play.xml")

    if "airbot_play" in mjcf_path:
        arm_dof = 6
    elif "arx_l5" in mjcf_path:
        arm_dof = 6
    elif "arx_x5" in mjcf_path:
        arm_dof = 6
    elif "piper" in mjcf_path:
        arm_dof = 6
    elif "rm65" in mjcf_path:
        arm_dof = 6
    elif "ur5e" in mjcf_path:
        arm_dof = 6
    elif "panda" in mjcf_path:
        arm_dof = 7
    elif "iiwa14" in mjcf_path:
        arm_dof = 7
    elif "xarm7" in mjcf_path:
        arm_dof = 7
    else:
        raise ValueError(f"Unsupported robot: {robot_name}")

    impedance_control = ("force" in mjcf_path)

    # 加载机器人模型的MJCF文件
    if not os.path.exists(mjcf_path):
        paths = [
            os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", mjcf_path),
            os.path.join(DISCOVERSE_ASSETS_DIR, mjcf_path),
            os.path.join(os.getcwd(), mjcf_path),
        ]
        for path in paths:
            if os.path.exists(path) and os.path.isfile(path) and (path.endswith(".xml") or path.endswith(".mjb")):
                mjcf_path = path
                break
    if mjcf_path is None:
        raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")
    print("mjcf_path : " , mjcf_path)

    # 设置末端执行器目标（mocap）名称
    mocap_name = "target"
    mocap_box_name = mocap_name + "_box"

    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    try:
        mid = mj_model.body(mocap_name).mocapid[0]
        if mid == -1:
            raise KeyError(f"Mocap body '{mocap_name}' not found")
    except KeyError:
        # 生成mocap刚体XML元素
        mocap_body_element = generate_mocap_xml(mocap_name)
        # 将mocap刚体添加到模型中
        mj_model = add_mocup_body_to_mjcf(mjcf_path, [mocap_body_element])
    mj_data = mujoco.MjData(mj_model)

    mocap_id = mj_model.body(mocap_name).mocapid[0]
    joint_q_sensor_idx = np.array(list(get_sensor_idx(mj_model, [f"joint{i+1}_pos" for i in range(arm_dof)]).values()))
    joint_dq_sensor_idx = np.array(list(get_sensor_idx(mj_model, [f"joint{i+1}_vel" for i in range(arm_dof)]).values()))
    joint_tau_sensor_idx = np.array(list(get_sensor_idx(mj_model, [f"joint{i+1}_torque" for i in range(arm_dof)]).values()))
    eef_sensor_idx = get_sensor_idx(mj_model, ["endpoint_pos", "endpoint_quat", "endpoint_vel", "endpoint_gyro", "endpoint_acc"])

    if impedance_control:
        mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "manipulator", f"robot_{robot_name}.xml")
        mj_model_idc = mujoco.MjModel.from_xml_path(mjcf_path)
        kp = [100, 100, 100, 5, 100, 5]
        kd = [5, 5, 5, 0.5, 0.5, 0.1]
        ID_controller = ImpedanceController(mj_model_idc, kp, kd)
    else:
        ID_controller = None

    if enable_record:
        camera_names = args.camera_names

        if len(camera_names):
            renderer = mujoco.Renderer(mj_model)
            img_width, img_height = 640, 480
            renderer._width = img_width
            renderer._height = img_height
            renderer._rect.width = img_width
            renderer._rect.height = img_height
        else:
            renderer = None

        def get_observation():
            tmat_target = get_mocap_tmat(mj_data, mocap_id)
            tmat_arm_base = get_site_tmat(mj_data, "armbase")
            tmat_target_local = np.linalg.inv(tmat_arm_base) @ tmat_target
            target_position = tmat_target_local[:3, 3]
            target_quat = Rotation.from_matrix(tmat_target_local[:3, :3]).as_quat()[[3,0,1,2]]
            obs = {
                "time": mj_data.time,
                "jq"  : mj_data.sensordata[joint_q_sensor_idx].tolist(),
                "jv"  : mj_data.sensordata[joint_dq_sensor_idx].tolist(),
                "tau" : mj_data.sensordata[joint_tau_sensor_idx].tolist(),
                "eef_pos" : mj_data.sensordata[eef_sensor_idx["endpoint_pos"]:eef_sensor_idx["endpoint_pos"]+3].tolist(),
                "eef_quat": mj_data.sensordata[eef_sensor_idx["endpoint_quat"]:eef_sensor_idx["endpoint_quat"]+4].tolist(),
                "eef_vel" : mj_data.sensordata[eef_sensor_idx["endpoint_vel"]:eef_sensor_idx["endpoint_vel"]+3].tolist(),
                "eef_gyro": mj_data.sensordata[eef_sensor_idx["endpoint_gyro"]:eef_sensor_idx["endpoint_gyro"]+3].tolist(),
                "eef_acc" : mj_data.sensordata[eef_sensor_idx["endpoint_acc"]:eef_sensor_idx["endpoint_acc"]+3].tolist(),
                "end_force" : ID_controller.get_ext_force().tolist() if impedance_control else [],
                "action" : np.hstack([target_position, target_quat]).tolist(),
                "img" : {}
            }
            if renderer:
                for camera_name in camera_names:
                    renderer.update_scene(mj_data, camera_name)
                    obs["img"][camera_name] = renderer.render()
            return obs

    # Create a Mink configuration
    configuration = mink.Configuration(mj_model)
    # Define tasks
    end_effector_task = mink.FrameTask(
        frame_name="endpoint",
        frame_type="site",
        position_cost=100.0,
        orientation_cost=10.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=mj_model, cost=1e-2)
    mink_tasks = [end_effector_task, posture_task]
    
    try:
        def key_press_callback(key):
            pass

        # 启动MuJoCo查看器
        viewer = mujoco.viewer.launch_passive(
            mj_model, mj_data, 
            show_left_ui=False, 
            show_right_ui=False,
            key_callback=key_press_callback
        )
    except RuntimeError as e:
        if "mjpython" in str(e) and platform.system() == "Darwin":
            print("\n错误: 在macOS上必须使用mjpython运行此脚本")
            print("请使用以下命令:")
            print(f"mjpython {' '.join(sys.argv)}")
        else:
            print(f"运行时错误: {e}")
        sys.exit(1)

    try:
        # 设置渲染帧率
        render_fps = 125.0
        # 计算渲染间隔，确保按照指定帧率渲染
        render_gap = int(1.0 / render_fps / mj_model.opt.timestep)

        obs_lst = []
        camera_encoders = {}
        total_record_cnt = 0
        save_dir = ""
        success = False

        if not args.hide_mocap:
            viewer.opt.geomgroup[5] = 1  # 显示group 5中的几何体

        def reset():
            global obs_lst, camera_encoders, save_dir, success

            mujoco.mj_resetDataKeyframe(mj_model, mj_data, mj_model.key(0).id)
            mujoco.mj_forward(mj_model, mj_data)
            mink.move_mocap_to_frame(mj_model, mj_data, mocap_name, "endpoint", "site")
            configuration.update(mj_data.qpos)
            posture_task.set_target_from_configuration(configuration)

            save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", f"{robot_name}_{task_name}", f"{total_record_cnt:03d}")

            if enable_record:
                obs_lst = []
                camera_encoders = {}
                if len(camera_names):
                    for cam_name in camera_names:
                        if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name) < 0:
                            print(f"Camera '{cam_name}' not found in the MJCF model.")
                        else:
                            camera_encoders[cam_name] = PyavImageEncoder(
                                img_width,
                                img_height,
                                save_dir,
                                cam_name,
                                fps=args.record_frequency or 24,
                            )
            success = False

        reset()
        last_select = viewer.perturb.select
        last_mj_time = mj_data.time
        while viewer.is_running():
            # 记录步骤开始时间
            step_start = time.time()
            if success and enable_record:
                total_record_cnt += 1
                recoder_state_file(save_dir, obs_lst)
                mj_data.time = -1

            if last_mj_time > mj_data.time:
                # after reset signal
                if enable_record and len(camera_names):
                    for ec in camera_encoders.values():
                        ec.close()
                reset()

            if enable_record and len(obs_lst) <= mj_data.time * args.record_frequency:
                obs = get_observation()
                imgs = obs.pop('img')
                for cam_id, img in imgs.items():
                    camera_encoders[cam_id].encode(img, obs["time"])
                obs_lst.append(obs)

            ###################################################################################
            if task_name == "peg_in_hole":
                tmat_endpoint = get_site_tmat(mj_data, "endpoint")
                tmat_hole = get_body_tmat(mj_data, "hole")
                dist_xy = np.linalg.norm(tmat_endpoint[:2,3] - tmat_hole[:2,3])
                dist_z = tmat_endpoint[2,3] - tmat_hole[2,3]
                if dist_xy < 0.005 and dist_z < 0.001 and np.abs(tmat_endpoint[0,2]) > 0.9996573249755573:
                    mj_model.geom("peg_box").rgba = (0, 1, 0, 1)
                    success = True
                else:
                    mj_model.geom("peg_box").rgba = (1, 0, 0, 1)

            ###################################################################################

            if args.mouse_3d:
                state = pyspacemouse.read()
                delta_position = np.array([state.y, -state.x, state.z])
                delta_position *= (np.abs(delta_position) > 0.01)
                delta_position = np.pow(delta_position, 2) * np.sign(delta_position)
                delta_euler = np.array([-state.roll, -state.pitch, state.yaw])

                # tmat_mocap = get_mocap_tmat(mj_data, mocap_id)
                rmat_base = get_site_tmat(mj_data, "armbase")[:3,:3]
                mj_data.mocap_pos[mocap_id] += (rmat_base @ delta_position) * 0.2 / render_fps

            T_wt = mink.SE3.from_mocap_name(mj_model, mj_data, mocap_name)
            end_effector_task.set_target(T_wt)
            res = converge_ik(
                configuration,
                mink_tasks,
                mj_model.opt.timestep,
                SOLVER,
                POS_THRESHOLD,
                ORI_THRESHOLD,
                MAX_ITERS,
            )
            if impedance_control:
                q_desired = configuration.q[:arm_dof]
                ID_controller.set_target(q_desired)
                ID_controller.update_state(mj_data.qpos[:arm_dof], mj_data.qvel[:arm_dof])
            else:
                mj_data.ctrl[:arm_dof] = configuration.q[:arm_dof]

            if res:
                # 设置目标框为绿色（表示IK计算成功）
                mj_model.geom(mocap_box_name).rgba = (0.3, 0.6, 0.3, 0.2)
            else:
                # 设置目标框为红色（表示IK计算失败）
                mj_model.geom(mocap_box_name).rgba = (0.6, 0.3, 0.3, 0.2)

            # 执行渲染间隔次数的物理仿真步骤
            for i in range(render_gap):
                if impedance_control: 
                    torque = ID_controller.compute_torque()
                    mj_data.ctrl[:arm_dof] = torque
                    ID_controller.update_state(mj_data.qpos[:arm_dof], mj_data.qvel[:arm_dof], torque)
                    ee_force = ID_controller.get_ext_force()
                # 执行物理仿真步骤
                mujoco.mj_step(mj_model, mj_data)
            last_mj_time = mj_data.time

            # 同步查看器状态
            viewer.sync()

            if last_select != viewer.perturb.select:
                body_name = mj_model.body(viewer.perturb.select).name
                print(f"选择物体: {body_name}")
                tmat_body = get_body_tmat(mj_data, body_name)
                if body_name != mocap_name:
                    print(">>> select object:")
                    print(tmat_body)
                print(">>> target_body: ")
                print(T_wt.as_matrix())
                print(">>> object2target")
                print(np.linalg.inv(tmat_body) @ T_wt.as_matrix())
                last_select = viewer.perturb.select

            # 计算下一步开始前需要等待的时间，保证帧率稳定
            time_until_next_step = (1. / render_fps) - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        print("用户中断，退出程序")

    finally:
        viewer.close()
        if args.mouse_3d:
            pyspacemouse.close()
        if enable_record and len(camera_names):
            for ec in camera_encoders.values():
                try:
                    ec.close()
                except Exception:
                    pass
        print("程序结束")