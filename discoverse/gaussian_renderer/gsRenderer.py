import os
import torch
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

from discoverse.gaussian_renderer import util_gau
from discoverse.gaussian_renderer.renderer_cuda import CUDARenderer

from discoverse import DISCOVERSE_ASSETS_DIR

class GSRenderer:
    """
    GSRenderer 用于管理高斯点云模型的加载、相机参数设置以及渲染流程。

    属性:
        width (int): 渲染宽度。
        height (int): 渲染高度。
        camera (util_gau.Camera): 相机对象，包含内参和分辨率。
        update_gauss_data (bool): 标记高斯数据是否需要更新。
        scale_modifier (float): 高斯点云缩放因子。
        renderer (CUDARenderer): CUDA 渲染器实例。
        camera_tran (np.ndarray): 当前相机平移向量。
        camera_quat (np.ndarray): 当前相机四元数旋转。
        gaussians_all (dict): 所有加载的高斯点云数据。
        gaussians_idx (dict): 每个对象在全局高斯数组中的起始索引。
        gaussians_size (dict): 每个对象的高斯点数量。
    """

    def __init__(self, models_dict:dict, render_width=1920, render_height=1080):
        """
        初始化 GSRenderer，加载所有高斯点云模型，并初始化渲染器和相机。

        参数:
            models_dict (dict): 包含模型名称与文件名的字典，需包含 'background' 键。
            render_width (int): 渲染宽度，默认1920。
            render_height (int): 渲染高度，默认1080。
        """
        self.width = render_width
        self.height = render_height

        self.camera = util_gau.Camera(self.height, self.width)

        self.update_gauss_data = False

        self.scale_modifier = 1.

        self.renderer = CUDARenderer(self.camera.w, self.camera.h)
        self.camera_tran = np.zeros(3)
        self.camera_quat = np.zeros(4)

        self.gaussians_all:dict[util_gau.GaussianData] = {}
        self.gaussians_idx = {}
        self.gaussians_size = {}
        idx_sum = 0

        gs_model_dir = Path(os.path.join(DISCOVERSE_ASSETS_DIR, "3dgs"))

        bg_key = "background"
        data_path = Path(os.path.join(gs_model_dir, models_dict[bg_key]))
        gs = util_gau.load_ply(data_path)
        if "background_env" in models_dict.keys():
            bgenv_key = "background_env"
            bgenv_gs = util_gau.load_ply(Path(os.path.join(gs_model_dir, models_dict[bgenv_key])))
            gs.xyz = np.concatenate([gs.xyz, bgenv_gs.xyz], axis=0)
            gs.rot = np.concatenate([gs.rot, bgenv_gs.rot], axis=0)
            gs.scale = np.concatenate([gs.scale, bgenv_gs.scale], axis=0)
            gs.opacity = np.concatenate([gs.opacity, bgenv_gs.opacity], axis=0)
            gs.sh = np.concatenate([gs.sh, bgenv_gs.sh], axis=0)

        self.gaussians_all[bg_key] = gs
        self.gaussians_idx[bg_key] = idx_sum
        self.gaussians_size[bg_key] = gs.xyz.shape[0]
        idx_sum = self.gaussians_size[bg_key]

        for i, (k, v) in enumerate(models_dict.items()):
            if k != "background" and k != "background_env":
                data_path = Path(os.path.join(gs_model_dir, v))
                gs = util_gau.load_ply(data_path)
                self.gaussians_all[k] = gs
                self.gaussians_idx[k] = idx_sum
                self.gaussians_size[k] = gs.xyz.shape[0]
                idx_sum += self.gaussians_size[k]

        self.update_activated_renderer_state(self.gaussians_all)

        for name in self.gaussians_all.keys():
            # :TODO: 找到哪里被改成torch了
            try:
                self.gaussians_all[name].R = self.gaussians_all[name].R.numpy()
            except:
                pass

    def update_camera_intrin_lazy(self):
        """
        若相机内参已被修改，则更新渲染器中的相机内参，并重置脏标记。
        """
        if self.camera.is_intrin_dirty:
            self.renderer.update_camera_intrin(self.camera)
            self.camera.is_intrin_dirty = False

    def update_activated_renderer_state(self, gaus: util_gau.GaussianData):
        """
        更新渲染器的高斯点云数据、缩放因子、相机参数和分辨率。

        参数:
            gaus (util_gau.GaussianData): 当前激活的高斯点云数据。
        """
        self.renderer.update_gaussian_data(gaus)
        self.renderer.set_scale_modifier(self.scale_modifier)
        self.renderer.update_camera_pose(self.camera)
        self.renderer.update_camera_intrin(self.camera)
        self.renderer.set_render_reso(self.camera.w, self.camera.h)

    def set_obj_pose(self, obj_name, trans, quat_wzyx):
        """
        设置指定对象的位姿（平移和旋转），并同步到渲染器。

        参数:
            obj_name (str): 对象名称。
            trans (np.ndarray): 平移向量。
            quat_wzyx (np.ndarray): 四元数旋转（WZYX顺序）。
        """
        if not ((self.gaussians_all[obj_name].origin_rot == quat_wzyx).all() and (self.gaussians_all[obj_name].origin_xyz == trans).all()):
            self.update_gauss_data = True
            self.gaussians_all[obj_name].origin_rot = quat_wzyx.copy()
            self.gaussians_all[obj_name].origin_xyz = trans.copy()
            self.renderer.gau_xyz_all_cu[self.gaussians_idx[obj_name]:self.gaussians_idx[obj_name]+self.gaussians_size[obj_name],:] = torch.from_numpy(trans).cuda().requires_grad_(False)
            self.renderer.gau_rot_all_cu[self.gaussians_idx[obj_name]:self.gaussians_idx[obj_name]+self.gaussians_size[obj_name],:] = torch.from_numpy(quat_wzyx).cuda().requires_grad_(False)

    def set_camera_pose(self, trans, quat_xyzw):
        """
        设置相机的位姿（平移和旋转），并同步到渲染器。

        参数:
            trans (np.ndarray): 相机平移向量。
            quat_xyzw (np.ndarray): 相机四元数旋转（XYZW顺序）。
        """
        if not ((self.camera_tran == trans).all() and (self.camera_quat == quat_xyzw).all()):
            self.camera_tran[:] = trans[:]
            self.camera_quat[:] = quat_xyzw[:]
            rmat = Rotation.from_quat(quat_xyzw).as_matrix()
            self.renderer.update_camera_pose_from_topic(self.camera, rmat, trans)

    def set_camera_fovy(self, fovy):
        """
        设置相机的视场角（fovy），如有变化则更新。

        参数:
            fovy (float): 新的视场角。
        """
        if not fovy == self.camera.fovy:
            self.camera.update_fovy(fovy)
    
    def set_camera_resolution(self, height, width):
        """
        设置相机和渲染器的分辨率，如有变化则更新。

        参数:
            height (int): 新的高度。
            width (int): 新的宽度。
        """
        if not (height == self.camera.h and width == self.camera.w):
            self.camera.update_resolution(height, width)
            self.renderer.set_render_reso(width, height)

    def render(self, render_depth=False):
        """
        执行渲染操作，返回渲染结果。

        参数:
            render_depth (bool): 是否渲染深度图，默认为 False。

        返回:
            渲染结果（通常为图像或深度图）。
        """
        self.update_camera_intrin_lazy()
        return self.renderer.draw(render_depth)