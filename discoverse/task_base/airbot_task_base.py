import os
import json
import fractions
import av.video
import glfw
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from discoverse.robots_env.airbot_play_base import AirbotPlayBase, AirbotPlayCfg
import av

# 定义使用 pyav 库进行视频编码的类
class PyavImageEncoder:
    """
    该类用于将图像编码成视频文件，利用 pyav 库进行视频封装。
    """

    def __init__(self, width, height, save_path, id):
        """
        初始化视频编码器

        参数:
            width (int): 视频宽度
            height (int): 视频高度
            save_path (str): 保存视频的目录
            id (int): 相机标识号，用于命名视频文件
        """
        self.width = width
        self.height = height
        # 拼接得到视频文件路径
        self.av_file_path = os.path.join(save_path, f"cam_{id}.mp4")
        # 若文件已存在，则先删除
        if os.path.exists(self.av_file_path):
            os.remove(self.av_file_path)
        # 打开输出容器，设置格式为 mp4
        container = av.open(self.av_file_path, "w", format="mp4")
        # 添加视频流，编码格式为 h264，加快预设选项为 fast
        stream: av.video.stream.VideoStream = container.add_stream("h264", options={"preset": "fast"})
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        self._time_base = int(1e6)
        # 设置视频流的时间基准
        stream.time_base = fractions.Fraction(1, self._time_base)
        self.container = container
        self.stream = stream
        self.start_time = None
        self.last_time = None
        self._cnt = 0

    def encode(self, image: np.ndarray, timestamp: float):
        """
        将单帧图像编码并写入视频文件

        参数:
            image (np.ndarray): RGB 图像数据
            timestamp (float): 当前帧的时间戳
        """
        self._cnt += 1
        # 如果是第一帧，则记录起始时间
        if self.start_time is None:
            self.start_time = timestamp
            self.last_time = 0
            self.container.metadata["comment"] = str({"base_stamp": int(self.start_time * self._time_base)})
        # 从 numpy 数组构建视频帧，格式为 rgb24
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        cur_time = timestamp
        # 计算 pts（显示时间戳）
        frame.pts = int((cur_time - self.start_time) * self._time_base)
        frame.time_base = self.stream.time_base
        # 确保时间戳单调递增
        assert cur_time > self.last_time, f"Time error: {cur_time} <= {self.last_time}"
        self.last_time = cur_time
        # 编码视频帧，并写入容器
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        """
        编码剩余帧并关闭视频文件
        """
        # 清空视频编码器中的缓冲帧并将其写入文件
        if self.container is not None:
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
        self.container = None

    def remove_av_file(self):
        """
        删除生成的视频文件
        """
        if os.path.exists(self.av_file_path):
            os.remove(self.av_file_path)
            print(f">>>>> Removed {self.av_file_path}")

def recoder_airbot_play(save_path, act_lst, obs_lst, cfg: AirbotPlayCfg):
    """
    保存仿真过程中采集的观测数据与动作到 JSON 文件中

    参数:
        save_path (str): 保存数据的目录
        act_lst (list): 动作列表
        obs_lst (list): 观测列表
        cfg (AirbotPlayCfg): 仿真配置对象
    """
    os.makedirs(save_path, exist_ok=True)

    # 保存 JSON 数据，其中包含时间、关节位置信息及动作列表
    with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
        time = []
        jq = []
        for obs in obs_lst:
            time.append(obs['time'])
            jq.append(obs['jq'])
        json.dump({
            "time" : time,
            "obs"  : {
                "jq" : jq,
            },
            "act"  : act_lst,
        }, fp)
    # 可选: 打印保存提示信息
    # print(f"Saved data to {save_path}")

# 定义仿真任务基础类，继承自 AirbotPlayBase
class AirbotPlayTaskBase(AirbotPlayBase):
    """
    该类扩展了 AirbotPlayBase，增加了一些任务相关的功能，如域随机化、动作判定以及额外的键盘交互逻辑。
    """
    # 目标控制信号（初始为全零）
    target_control = np.zeros(7)
    # 关节运动比例，初始为全零
    joint_move_ratio = np.zeros(7)
    # 动作执行情况记录字典
    action_done_dict = {
        "joint"   : False,
        "gripper" : False,
        "delay"   : False,
    }
    # 延迟计数器
    delay_cnt = 0
    # 重置信号标识，用于标记当前是否重置
    reset_sig = False
    # 相机标识号
    cam_id = 0

    def resetState(self):
        """
        重置仿真状态，调用父类的 resetState 方法，并执行域随机化设置
        """
        super().resetState()
        # 将目标控制重置为初始控制信号
        self.target_control[:] = self.init_joint_ctrl[:]
        # 执行域随机化（环境参数随机化，用于提升模型的泛化能力）
        self.domain_randomization()
        # 再次执行前向动力学更新
        mujoco.mj_forward(self.mj_model, self.mj_data)
        # 标记已重置
        self.reset_sig = True

    def random_table_height(self, table_name="table", obj_name_list=[]):
        """
        随机调整桌子高度，同时调整相关对象位置

        参数:
            table_name (str): 桌子的名称
            obj_name_list (list): 与桌子关联的物体名称列表
        """
        # 如果还没有记录桌子的初始位置，则记录下来
        if not hasattr(self, "table_init_posi"):
            self.table_init_posi = self.mj_model.body(table_name).pos.copy()
        # 随机生成高度偏移量（0~0.1之间）
        change_height = np.random.uniform(0, 0.1)
        # 先将桌子位置重置为初始位置
        self.mj_model.body(table_name).pos = self.table_init_posi.copy()
        # 修改高度（通常为 z 轴，即索引 2）
        self.mj_model.body(table_name).pos[2] = self.table_init_posi[2] - change_height
        # 对列表中每个物体也相应调低高度
        for obj_name in obj_name_list:
            self.object_pose(obj_name)[2] -= change_height

    def random_table_texture(self):
        """
        随机更换桌子纹理，通过更新材质纹理和材质属性实现视觉随机化
        """
        self.update_texture("tc_texture", self.get_random_texture())
        self.random_material("tc_texture")

    def random_material(self, mtl_name, random_color=False, emission=False):
        """
        随机调整材质参数，用于生成随机化的材质效果

        参数:
            mtl_name (str): 材质名称
            random_color (bool): 是否随机颜色
            emission (bool): 是否随机自发光属性
        """
        try:
            if random_color:
                # 随机设定颜色（RGB三个分量）
                self.mj_model.material(mtl_name).rgba[:3] = np.random.rand(3)
            if emission:
                # 随机设定发光强度
                self.mj_model.material(mtl_name).emission = np.random.rand()
            # 分别随机设定高光、反射率和光泽度
            self.mj_model.material(mtl_name).specular = np.random.rand()
            self.mj_model.material(mtl_name).reflectance = np.random.rand()
            self.mj_model.material(mtl_name).shininess = np.random.rand()
        except KeyError:
            print(f"Warning: material {mtl_name} not found")

    def random_light(self, random_dir=True, random_color=True, random_active=True, write_color=False):
        """
        随机调整光源属性，包括颜色、方向和激活状态

        参数:
            random_dir (bool): 是否随机光源方向
            random_color (bool): 是否随机光源颜色
            random_active (bool): 是否随机光源激活状态
            write_color (bool): 是否强制写入随机颜色到所有光源
        """
        if write_color:
            for i in range(self.mj_model.nlight):
                self.mj_model.light_ambient[i, :] = np.random.random()
                self.mj_model.light_diffuse[i, :] = np.random.random()
                self.mj_model.light_specular[i, :] = np.random.random()
        elif random_color:
            self.mj_model.light_ambient[...] = np.random.random(size=self.mj_model.light_ambient.shape)
            self.mj_model.light_diffuse[...] = np.random.random(size=self.mj_model.light_diffuse.shape)
            self.mj_model.light_specular[...] = np.random.random(size=self.mj_model.light_specular.shape)

        # 根据光源是否是定向光调整其属性，通常降低定向光的亮度
        if write_color or random_color:
            for i in range(self.mj_model.nlight):
                if self.mj_model.light_directional[i]:
                    self.mj_model.light_diffuse[i, :] *= 0.2
                    self.mj_model.light_ambient[i, :] *= 0.5
                    self.mj_model.light_specular[i, :] *= 0.5

        # 随机设定每个光源是否激活
        if random_active:
            self.mj_model.light_active[:] = np.int32(np.random.rand(self.mj_model.nlight) > 0.5).tolist()

        # 保证至少有一个光源激活
        if np.sum(self.mj_model.light_active) == 0:
            self.mj_model.light_active[np.random.randint(self.mj_model.nlight)] = 1

        # 随机调整光源位置（前两维添加噪声，第三维添加较小噪声）
        self.mj_model.light_pos[:,:2] = self.mj_model.light_pos0[:,:2] + np.random.normal(scale=0.3, size=self.mj_model.light_pos[:,:2].shape)
        self.mj_model.light_pos[:,2] = self.mj_model.light_pos0[:,2] + np.random.normal(scale=0.2, size=self.mj_model.light_pos[:,2].shape)

        # 随机调整光源方向，并规范化向量以保证其为单位向量
        if random_dir:
            self.mj_model.light_dir[:] = np.random.random(size=self.mj_model.light_dir.shape) - 0.5
            self.mj_model.light_dir[:,2] *= 2.0
            self.mj_model.light_dir[:] = self.mj_model.light_dir[:] / np.linalg.norm(self.mj_model.light_dir[:], axis=1, keepdims=True)
            # 保证光源 z 轴方向为负
            self.mj_model.light_dir[:,2] = -np.abs(self.mj_model.light_dir[:,2])

    def domain_randomization(self):
        """
        执行领域随机化，随机调整仿真环境中的各种物理和视觉参数
        用户可根据具体任务实现自定义的随机化逻辑
        """
        pass

    def checkActionDone(self):
        """
        判断当前动作是否已完成

        检查:
            - 关节动作是否达到目标值（关节位置误差小于一定阈值且速度很低）
            - 抓手动作是否达到目标值（误差和速度均在容差范围内）
            - 延时是否结束

        返回:
            bool: 如果所有条件满足，动作标记为完成
        """
        joint_done = np.allclose(self.sensor_joint_qpos[:6], self.target_control[:6], atol=3e-2) and np.abs(self.sensor_joint_qvel[:6]).sum() < 0.1
        gripper_done = np.allclose(self.sensor_joint_qpos[6], self.target_control[6], atol=0.4) and np.abs(self.sensor_joint_qvel[6]).sum() < 0.125
        # 减少延迟计数器
        self.delay_cnt -= 1
        delay_done = (self.delay_cnt <= 0)
        # 更新动作完成状态记录字典
        self.action_done_dict = {
            "joint"   : joint_done,
            "gripper" : gripper_done,
            "delay"   : delay_done,
        }
        return joint_done and gripper_done and delay_done

    def printMessage(self):
        """
        打印当前仿真状态信息，包括关节状态、目标控制、动作状态以及相机信息
        """
        # 先调用父类打印基础仿真数据
        super().printMessage()
        print("    target control = ", self.target_control)
        print("    action done: ")
        for k, v in self.action_done_dict.items():
            print(f"        {k}: {v}")

        print("camera foyv = ", self.mj_model.vis.global_.fovy)
        # 获取当前相机位置和朝向（四元数表示），并转换为旋转矩阵进行展示
        cam_xyz, cam_wxyz = self.getCameraPose(self.cam_id)
        print(f"    camera_{self.cam_id} =\n({cam_xyz}\n{Rotation.from_quat(cam_wxyz[[1,2,3,0]]).as_matrix()})")

    def check_success(self):
        """
        判断任务是否成功
        该方法为抽象方法，需在子类中实现具体的成功判定逻辑
        """
        raise NotImplementedError

    def on_key(self, window, key, scancode, action, mods):
        """
        处理键盘按键事件，用于在仿真过程中调整视角参数

        参数:
            window: 当前窗口对象
            key: 按键编码
            scancode: 按键扫描码
            action: 按键动作状态（PRESS/RELEASE）
            mods: 其他修饰键状态

        返回:
            ret: 父类 on_key 方法的处理结果
        """
        ret = super().on_key(window, key, scancode, action, mods)
        if action == glfw.PRESS:
            # 按 "-" 键降低视野范围
            if key == glfw.KEY_MINUS:
                self.mj_model.vis.global_.fovy = np.clip(self.mj_model.vis.global_.fovy * 0.95, 5, 175)
            # 按 "=" 键增大视野范围
            elif key == glfw.KEY_EQUAL:
                self.mj_model.vis.global_.fovy = np.clip(self.mj_model.vis.global_.fovy * 1.05, 5, 175)
        return ret