from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from discoverse import DISCOVERSE_ASSETS_DIR
import mink

_XML = Path(DISCOVERSE_ASSETS_DIR) / "mjcf" / "mink_panda.xml"

# IK parameters
SOLVER = "quadprog"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 20

def converge_ik(
    configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters
):
    """
    Runs up to 'max_iters' of IK steps. Returns True if position and orientation
    are below thresholds, otherwise False.
    """
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3)
        configuration.integrate_inplace(vel, dt)

        # Only checking the first FrameTask here (end_effector_task).
        # If you want to check multiple tasks, sum or combine their errors.
        err = tasks[0].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

        if pos_achieved and ori_achieved:
            return True
    return False


def main():
    # Load model & data
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Create a Mink configuration
    configuration = mink.Configuration(model)

    # Define tasks
    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [end_effector_task, posture_task]

    # Initialize viewer in passive mode
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Reset simulation data to the 'home' keyframe
        mujoco.mj_resetDataKeyframe(model, data, model.key(0).id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        initial_target_position = data.mocap_pos[0].copy()

        # Move the mocap target to the end-effector's current pose
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        # We'll track time ourselves for a smoother trajectory
        # rate = RateLimiter(frequency=1000.0, warn=False)

        dt = 2e-3
        period = 3
        trigger_time = 0
        local_time = 0.0

        while viewer.is_running():
            # Update our local time
            local_time += dt

            if 0.1 < local_time % period and trigger_time < 1e-3:
                data.ctrl[7] = 0.04
                data.mocap_pos[0] = data.body("block").xpos + np.array([0, 0, 0.05])
                trigger_time = local_time%period + dt
                print("1" * 20)
                # Update the end effector task target from the mocap body
                T_wt = mink.SE3.from_mocap_name(model, data, "target")
                end_effector_task.set_target(T_wt)

                # Attempt to converge IK
                converge_ik(
                    configuration,
                    tasks,
                    dt,
                    SOLVER,
                    POS_THRESHOLD,
                    ORI_THRESHOLD,
                    MAX_ITERS,
                )

                # Set robot controls (first 7 dofs in your configuration)
                data.ctrl[:7] = configuration.q[:7]
            elif 1.0 < local_time % period and trigger_time < 1.0:
                data.mocap_pos[0] = data.body("block").xpos + np.array([0, 0, 0.02])
                trigger_time = local_time%period + dt
                print("2" * 20)
                # Update the end effector task target from the mocap body
                T_wt = mink.SE3.from_mocap_name(model, data, "target")
                end_effector_task.set_target(T_wt)

                # Attempt to converge IK
                converge_ik(
                    configuration,
                    tasks,
                    dt,
                    SOLVER,
                    POS_THRESHOLD,
                    ORI_THRESHOLD,
                    MAX_ITERS,
                )

                # Set robot controls (first 7 dofs in your configuration)
                data.ctrl[:7] = configuration.q[:7]
            elif 1.5 < local_time % period and trigger_time < 1.5:
                data.ctrl[7] = 0.01
                trigger_time = local_time% period + dt
                print("3" * 20)
                # Update the end effector task target from the mocap body
                T_wt = mink.SE3.from_mocap_name(model, data, "target")
                end_effector_task.set_target(T_wt)

                # Attempt to converge IK
                converge_ik(
                    configuration,
                    tasks,
                    dt,
                    SOLVER,
                    POS_THRESHOLD,
                    ORI_THRESHOLD,
                    MAX_ITERS,
                )

                # Set robot controls (first 7 dofs in your configuration)
                data.ctrl[:7] = configuration.q[:7]
            elif 2. < local_time % period and trigger_time < 2.:
                data.mocap_pos[0] = initial_target_position
                trigger_time = local_time%period + dt
                print("4" * 20)
                # Update the end effector task target from the mocap body
                T_wt = mink.SE3.from_mocap_name(model, data, "target")
                end_effector_task.set_target(T_wt)

                # Attempt to converge IK
                converge_ik(
                    configuration,
                    tasks,
                    dt,
                    SOLVER,
                    POS_THRESHOLD,
                    ORI_THRESHOLD,
                    MAX_ITERS,
                )

                # Set robot controls (first 7 dofs in your configuration)
                data.ctrl[:7] = configuration.q[:7]
            elif period-(dt*2) < local_time % period and trigger_time < period-(dt*2):
                data.ctrl[7] = 0.04
                trigger_time = 0
                print("5" * 20)
                # Update the end effector task target from the mocap body
                T_wt = mink.SE3.from_mocap_name(model, data, "target")
                end_effector_task.set_target(T_wt)

                # Attempt to converge IK
                converge_ik(
                    configuration,
                    tasks,
                    dt,
                    SOLVER,
                    POS_THRESHOLD,
                    ORI_THRESHOLD,
                    MAX_ITERS,
                )

                # Set robot controls (first 7 dofs in your configuration)
                data.ctrl[:7] = configuration.q[:7]
            # else:
            #     print(f"{local_time:.4f} {trigger_time:.4f}")

            # # Update the end effector task target from the mocap body
            # T_wt = mink.SE3.from_mocap_name(model, data, "target")
            # end_effector_task.set_target(T_wt)

            # # Attempt to converge IK
            # converge_ik(
            #     configuration,
            #     tasks,
            #     dt,
            #     SOLVER,
            #     POS_THRESHOLD,
            #     ORI_THRESHOLD,
            #     MAX_ITERS,
            # )

            # # Set robot controls (first 7 dofs in your configuration)
            # data.ctrl[:7] = configuration.q[:7]

            # Step simulation
            for _ in range(4):
                mujoco.mj_step(model, data)

            # Visualize at fixed FPS
            viewer.sync()
            # rate.sleep()

if __name__ == "__main__":
    main()
