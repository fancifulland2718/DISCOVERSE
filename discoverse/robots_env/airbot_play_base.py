import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig

# 定义自定义配置类，继承自 BaseConfig
class AirbotPlayCfg(BaseConfig):
    # mjcf 文件路径配置
    mjcf_file_path = "mjcf/airbot_play_floor.xml"
    # 数据降采样因子
    decimation     = 4
    # 仿真步长
    timestep       = 0.005
    # 是否启用同步模式
    sync           = True
    # 是否无头模式（不显示图形界面）
    headless       = False
    # 渲染设置：帧率、宽度和高度
    render_set     = {
        "fps"    : 30,
        "width"  : 1280,
        "height" : 720,
    }
    # 初始关节位置，使用 np.zeros 初始化七个自由度
    init_qpos = np.zeros(7)
    # 观察用的 RGB 相机 ID，此处未指定
    obs_rgb_cam_id  = None
    # 机器人中各个连杆名称列表
    rb_link_list   = ["arm_base", "link1", "link2", "link3", "link4", "link5", "link6", "right", "left"]
    # 环境中物体列表，此处为空列表
    obj_list       = []
    # 是否使用高斯渲染器（Gaussian renderer）
    use_gaussian_renderer = False
    # 高斯渲染模型字典，映射机器人各部分名称到对应的 ply 模型路径
    gs_model_dict = {
        "arm_base"  : "airbot_play/arm_base.ply",
        "link1"     : "airbot_play/link1.ply",
        "link2"     : "airbot_play/link2.ply",
        "link3"     : "airbot_play/link3.ply",
        "link4"     : "airbot_play/link4.ply",
        "link5"     : "airbot_play/link5.ply",
        "link6"     : "airbot_play/link6.ply",
        "left"      : "airbot_play/left.ply",
        "right"     : "airbot_play/right.ply",
    }

# 定义自定义仿真类，继承自 SimulatorBase
class AirbotPlayBase(SimulatorBase):
    def __init__(self, config: AirbotPlayCfg):
        # 模型的关节数量，此处为 7
        self.nj = 7
        # 调用父类构造函数完成初始化工作
        super().__init__(config)

    # 在加载 mjcf 文件后执行一些自定义初始化操作
    def post_load_mjcf(self):
        try:
            # 检查配置是否有初始化关节位置 init_qpos 属性且其不为空
            if hasattr(self.config, "init_qpos") and self.config.init_qpos is not None:
                # 确认初始关节位置的长度与关节数一致，否则引发断言错误
                assert len(self.config.init_qpos) == self.nj, "init_qpos length must match the number of joints"
                # 将配置中的初始关节位置转换为 numpy 数组
                self.init_joint_pose = np.array(self.config.init_qpos) # 这里有关节转换关系
                # 初始化关节控制数据，同初始关节位置一致
                self.init_joint_ctrl = self.init_joint_pose.copy()
            else:
                # 如果配置中未找到 init_qpos，则抛出 KeyError 异常
                raise KeyError("init_qpos not found in config")
        except KeyError as e:
            # 如果捕获到异常，则将关节位置及控制数据初始化为全零向量
            self.init_joint_pose = np.zeros(self.nj)
            self.init_joint_ctrl = np.zeros(self.nj)

        # 从传感器数据中截取出各部分数据：
        # 关节位置数据，关节速度，关节受力
        self.sensor_joint_qpos = self.mj_data.sensordata[:self.nj]
        self.sensor_joint_qvel = self.mj_data.sensordata[self.nj:2*self.nj]
        self.sensor_joint_force = self.mj_data.sensordata[2*self.nj:3*self.nj]
        # 末端执行器的局部位置信息
        self.sensor_endpoint_posi_local = self.mj_data.sensordata[3*self.nj:3*self.nj+3]
        # 末端执行器的局部四元数信息
        self.sensor_endpoint_quat_local = self.mj_data.sensordata[3*self.nj+3:3*self.nj+7]
        # 末端执行器的局部线速度
        self.sensor_endpoint_linear_vel_local = self.mj_data.sensordata[3*self.nj+7:3*self.nj+10]
        # 末端执行器的陀螺仪数据（角速度）
        self.sensor_endpoint_gyro = self.mj_data.sensordata[3*self.nj+10:3*self.nj+13]
        # 末端执行器的加速度数据
        self.sensor_endpoint_acc = self.mj_data.sensordata[3*self.nj+13:3*self.nj+16]

    # 打印仿真当前状态的信息（如时间、关节状态、传感器数据等）
    def printMessage(self):
        print("-" * 100)
        print("mj_data.time  = {:.3f}".format(self.mj_data.time))
        print("    arm .qpos  = {}".format(np.array2string(self.sensor_joint_qpos, separator=', ')))
        print("    arm .qvel  = {}".format(np.array2string(self.sensor_joint_qvel, separator=', ')))
        print("    arm .ctrl  = {}".format(np.array2string(self.mj_data.ctrl[:self.nj], separator=', ')))
        print("    arm .force = {}".format(np.array2string(self.sensor_joint_force, separator=', ')))
        # 打印末端执行器位置和欧拉角数据
        print("    sensor end posi  = {}".format(np.array2string(self.sensor_endpoint_posi_local, separator=', ')))
        print("    sensor end euler = {}".format(np.array2string(Rotation.from_quat(self.sensor_endpoint_quat_local[[1,2,3,0]]).as_euler("xyz"), separator=', ')))

    # 重置仿真状态，子类进一步延展其功能
    def resetState(self):
        # 调用 mujoco 提供的重置数据函数，将各项数据重置为初始状态
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        # 根据初始关节位置信息重置当前系统状态
        self.mj_data.qpos[:self.nj] = self.init_joint_pose.copy()
        # 根据初始关节控制信息重置控制输入
        self.mj_data.ctrl[:self.nj] = self.init_joint_ctrl.copy()
        # 调用前向动力学函数计算重置后的状态
        mujoco.mj_forward(self.mj_model, self.mj_data)

    # 根据输入动作更新控制信号
    def updateControl(self, action):
        # 若最后一个关节角度小于 0，则将其值置为 0
        if self.mj_data.qpos[self.nj-1] < 0.0:
            self.mj_data.qpos[self.nj-1] = 0.0
        # 对动作进行裁剪，确保其在控制信号允许的范围内，并更新控制信号
        self.mj_data.ctrl[:self.nj] = np.clip(action[:self.nj], self.mj_model.actuator_ctrlrange[:self.nj,0], self.mj_model.actuator_ctrlrange[:self.nj,1])

    # 检查仿真是否终止，目前始终返回 False
    def checkTerminated(self):
        return False

    # 获取当前仿真状态的观测值
    def getObservation(self):
        # 生成观测字典，包括时间、关节状态、受力信息、末端位置和四元数信息、RGB 图像及深度图像
        self.obs = {
            "time" : self.mj_data.time,
            "jq"   : self.sensor_joint_qpos.tolist(),
            "jv"   : self.sensor_joint_qvel.tolist(),
            "jf"   : self.sensor_joint_force.tolist(),
            "ep"   : self.sensor_endpoint_posi_local.tolist(),
            "eq"   : self.sensor_endpoint_quat_local.tolist(),
            "img"  : self.img_rgb_obs_s.copy(),
            "depth" : self.img_depth_obs_s.copy()
        }
        return self.obs

    # 获取特权（Privileged）信息，目前直接返回 getObservation 得到的观测数据
    def getPrivilegedObservation(self):
        return self.obs

    # 获取奖励信息，目前未定义具体奖励机制，返回 None
    def getReward(self):
        return None

# 模块运行入口，方便直接运行当前文件进行简单测试
if __name__ == "__main__":
    # 实例化配置
    cfg = AirbotPlayCfg()
    # 实例化仿真执行节点
    exec_node = AirbotPlayBase(cfg)

    # 重置仿真环境，获取初始观测数据
    obs = exec_node.reset()
    # 设置初始动作为初始关节位置
    action = exec_node.init_joint_pose[:exec_node.nj]
    # 循环执行仿真步骤，直至仿真停止
    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action)