import mujoco
import numpy as np

class ImpedanceController:
    def __init__(self, mj_model, kpl, kdl):
        self.mj_model = mj_model
        self.mj_data = mujoco.MjData(mj_model)
        self.Kp = np.diag(kpl)
        self.Kd = np.diag(kdl)
        self.ndof = len(kpl)
        self.q_desired = np.zeros(self.ndof)
        self.bias = np.zeros(mj_model.nv)

    def set_target(self, q_desired):
        self.q_desired = q_desired
    
    def update_state(self, q, dq, tau=None):
        self.q = q.copy()
        self.dq = dq.copy()
        self.tau = tau.copy() if tau is not None else None
        
        self.mj_data.qpos[:self.ndof] = q.copy()
        self.mj_data.qvel[:self.ndof] = dq.copy()
        if tau is not None:
            self.mj_data.ctrl[:len(tau)] = tau.copy()
   
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mujoco.mj_step(self.mj_model, self.mj_data)
    
    def get_ext_force(self):
        ee_body_name = "link6"
        ee_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
        jacp = np.zeros((3, self.mj_model.nv))   
        jacr = np.zeros((3, self.mj_model.nv))   
        point = np.zeros(3, dtype=np.float64)
        mujoco.mj_jac(self.mj_model, self.mj_data, jacp, jacr, point, ee_id)
        J = np.vstack((jacp[:, :self.mj_model.nu], jacr[:, :self.mj_model.nu])) # Jacobian Matrix
        mujoco.mj_inverse(self.mj_model, self.mj_data)
        tau_net = self.mj_data.qfrc_actuator[:self.mj_model.nu] - self.mj_data.qfrc_inverse[:self.mj_model.nu] # torque
        ee_force = np.linalg.pinv(J.T) @ tau_net
        return ee_force

    def compute_torque(self):      
        position_error = self.q_desired - self.q
        velocity_error = -self.dq
        mujoco.mj_rne(self.mj_model, self.mj_data, 0, self.bias)
        torque = self.Kp @ position_error + self.Kd @ velocity_error + self.bias[:self.mj_model.nu]
        return torque