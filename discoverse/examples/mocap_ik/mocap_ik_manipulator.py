import os
import sys
import time
import platform
import argparse
import mujoco
import mujoco.viewer
import numpy as np

import mink
import discoverse
from discoverse.envs import make_env
from discoverse.examples.mocap_ik.mocap_ik_utils import \
    add_mocup_body_to_mjcf, \
    generate_mocap_xml

from discoverse import DISCOVERSE_ASSETS_DIR

SOLVER = "quadprog"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 50
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
        "-r",
        "--robot",
        type=str,
        default=None,
        help="输入机器人模型名称",
        choices=["airbot_play", "arx_l5", "arx_x5", "iiwa14", "panda", "piper", "rm65", "ur5e", "xarm7"],
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default=None,
        help="输入目标名称",
        choices=["block_bridge_place", "close_laptop", "cover_cup", "open_drawer", "pick_jujube", "place_block", "place_coffeecup", "place_jujube", "place_jujube_coffeecup", "place_kiwi_fruit", "push_mouse", "stack_block"],
    )
    parser.add_argument(
        "-m",
        "--mjcf",
        type=str,
        default=None,
        help="输入MJCF文件的路径（可选）。如未指定，则使用默认的robot_airbot_play.xml"
    )

    # 检查是否在macOS上运行并给出适当的提示
    if platform.system() == "Darwin":
        print("\n===================================================")
        print("注意: 在macOS上运行MuJoCo查看器需要使用mjpython")
        print("请使用以下命令运行此脚本:")
        print(f"mjpython {' '.join(sys.argv)}")
        print("===================================================\n")
        
        user_input = input("是否继续尝试启动查看器? (y/n): ")
        if user_input.lower() != 'y':
            print("退出程序。")
            sys.exit(0)
    # 打印MuJoCo查看器的使用提示信息
    print("\n===================== MuJoCo 查看器使用说明 =====================")
    print("1. 双击选中机械臂末端的绿色方块，按下ctrl和鼠标左键，拖动鼠标可以旋转机械臂")
    print("   按下ctrl和鼠标右键，拖动鼠标可以平移机械臂末端")
    print("2. 按 Tab 键切换左侧 UI 的可视化界面。")
    print("   按 Shift+Tab 键切换右侧 UI 的可视化界面。")
    print("3. 在左侧 UI 中点击 'Copy State' 可以当前的机器人状态复制到剪贴板。")
    print("4. 在右侧 UI 的 control 面板中可以调节 gripper 滑块控制夹爪的开合。")
    print("================================================================\n")
    # 设置numpy输出格式
    np.set_printoptions(precision=5, suppress=True, linewidth=500)

    args = parser.parse_args()
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

    # 生成mocap刚体XML
    mocap_body_xml = generate_mocap_xml(mocap_name)
    # 将mocap刚体添加到模型中
    mj_model = add_mocup_body_to_mjcf(mjcf_path, mocap_body_xml, keep_tmp_xml=True)
    mj_data = mujoco.MjData(mj_model)

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
        # 启动MuJoCo查看器
        with mujoco.viewer.launch_passive(
            mj_model, mj_data, show_left_ui=False, show_right_ui=False
        ) as viewer:

            # 设置渲染帧率
            render_fps = 50
            # 计算渲染间隔，确保按照指定帧率渲染
            render_gap = int(1.0 / render_fps / mj_model.opt.timestep)

            # Initialize configuration and set posture task target
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, mj_model.key(0).id)
            configuration.update(mj_data.qpos)
            posture_task.set_target_from_configuration(configuration)
            mujoco.mj_forward(mj_model, mj_data)

            # Move the mocap target to the end-effector's current pose
            mink.move_mocap_to_frame(mj_model, mj_data, mocap_name, "endpoint", "site")

            while viewer.is_running():
                # 记录步骤开始时间
                step_start = time.time()

                T_wt = mink.SE3.from_mocap_name(mj_model, mj_data, "target")
                end_effector_task.set_target(T_wt)
                res = converge_ik(
                    configuration,
                    mink_tasks,
                    2e-3,
                    SOLVER,
                    POS_THRESHOLD,
                    ORI_THRESHOLD,
                    MAX_ITERS,
                )
                mj_data.ctrl[:arm_dof] = configuration.q[:arm_dof]

                if res:
                    # 设置目标框为绿色（表示IK计算成功）
                    mj_model.geom(mocap_box_name).rgba = (0.3, 0.6, 0.3, 0.2)
                else:
                    # 设置目标框为红色（表示IK计算失败）
                    mj_model.geom(mocap_box_name).rgba = (0.6, 0.3, 0.3, 0.2)

                # 执行渲染间隔次数的物理仿真步骤
                for _ in range(render_gap):
                    # 执行物理仿真步骤
                    mujoco.mj_step(mj_model, mj_data)

                # 同步查看器状态
                viewer.sync()
                
                # 计算下一步开始前需要等待的时间，保证帧率稳定
                time_until_next_step = (1. / render_fps) - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except RuntimeError as e:
        if "mjpython" in str(e) and platform.system() == "Darwin":
            print("\n错误: 在macOS上必须使用mjpython运行此脚本")
            print("请使用以下命令:")
            print(f"mjpython {' '.join(sys.argv)}")
        else:
            print(f"运行时错误: {e}")