import mujoco
import numpy as np

class ImpedanceController:
    """
    TBD: @LiZheng
    """
    def __init__(self, mj_model, kpl, kdl):
        self.mj_model = mj_model
        self.mj_data = mujoco.MjData(mj_model)
        self.Kp = np.diag(kpl)
        self.Kd = np.diag(kdl)
        self.ndof = len(kpl)
        self.q_desired = np.zeros(self.ndof)

    def set_target(self, q_desired):
        self.q_desired = q_desired
    
    def update_state(self, q, dq, tau=None):
        self.q = q
        self.dq = dq
        self.tau = tau
    
    def get_ext_force(self):
        raise NotImplementedError

    def compute_torque(self):
        position_error = self.q_desired - self.q
        velocity_error = -self.dq
        torque = self.Kp @ position_error + self.Kd @ velocity_error
        return torque