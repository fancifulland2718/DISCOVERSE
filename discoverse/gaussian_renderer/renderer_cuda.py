'''
Part of the code (CUDA and OpenGL memory transfer) is derived from https://github.com/jbaron34/torchwindow/tree/master
'''

from discoverse.gaussian_renderer import util_gau
import numpy as np
import torch
from dataclasses import dataclass
from diff_gaussian_rasterization import GaussianRasterizer as GausssianRasterizer_3d

@dataclass
class GaussianDataCUDA:
    """
    存储高斯点云的CUDA张量数据结构。

    属性:
        xyz (torch.Tensor): 高斯点的三维坐标 (N, 3)。
        rot (torch.Tensor): 高斯点的旋转四元数 (N, 4)。
        scale (torch.Tensor): 高斯点的缩放参数 (N, 3)。
        opacity (torch.Tensor): 高斯点的不透明度 (N, 1)。
        sh (torch.Tensor): 球谐系数 (N, D, 3)。
    """
    xyz: torch.Tensor
    rot: torch.Tensor
    scale: torch.Tensor
    opacity: torch.Tensor
    sh: torch.Tensor
    
    def __len__(self):
        """
        返回高斯点的数量。
        """
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        """
        返回球谐系数的维度。
        """
        return self.sh.shape[-2]

@dataclass
class GaussianRasterizationSettingsStorage:
    """
    存储高斯点云光栅化的渲染设置。

    属性:
        image_height (int): 输出图像高度。
        image_width (int): 输出图像宽度。
        tanfovx (float): 水平视场角的正切值。
        tanfovy (float): 垂直视场角的正切值。
        bg (torch.Tensor): 背景颜色 (3,)。
        scale_modifier (float): 缩放因子。
        viewmatrix (torch.Tensor): 视图矩阵 (4, 4)。
        projmatrix (torch.Tensor): 投影矩阵 (4, 4)。
        sh_degree (int): 球谐展开的阶数。
        campos (torch.Tensor): 相机位置 (3,)。
        prefiltered (bool): 是否预滤波。
        debug (bool): 是否开启调试模式。
    """
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

def gaus_cuda_from_cpu(gau: util_gau) -> GaussianDataCUDA:
    """
    将CPU上的高斯点云数据转换为CUDA张量。

    参数:
        gau (util_gau.GaussianData): CPU上的高斯点云数据。

    返回:
        GaussianDataCUDA: 包含CUDA张量的高斯点云数据。
    """
    gaus =  GaussianDataCUDA(
        xyz = torch.tensor(gau.xyz).float().cuda().requires_grad_(False),
        rot = torch.tensor(gau.rot).float().cuda().requires_grad_(False),
        scale = torch.tensor(gau.scale).float().cuda().requires_grad_(False),
        opacity = torch.tensor(gau.opacity).float().cuda().requires_grad_(False),
        sh = torch.tensor(gau.sh).float().cuda().requires_grad_(False)
    )
    gaus.sh = gaus.sh.reshape(len(gaus), -1, 3).contiguous()
    return gaus

class CUDARenderer:
    """
    基于CUDA的高斯点云渲染器，负责管理高斯点数据、相机参数和渲染流程。
    """
    def __init__(self, w, h):
        """
        初始化渲染器，设置渲染分辨率和默认参数。

        参数:
            w (int): 渲染宽度。
            h (int): 渲染高度。
        """
        super().__init__()
        raster_settings = {
            "image_height": int(h),
            "image_width": int(w),
            "tanfovx": 1,
            "tanfovy": 1,
            "bg": torch.Tensor([0., 0., 0]).float().cuda(),
            "scale_modifier": 1.,
            "viewmatrix": None,
            "projmatrix": None,
            "sh_degree": 3,
            "campos": None,
            "prefiltered": False,
            "debug": False
        }
        self.raster_settings = GaussianRasterizationSettingsStorage(**raster_settings)

        self.depth_render = False
        self.need_rerender = True
        self.render_rgb_img = None
        self.render_depth_img = None

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        """
        更新高斯点云数据到CUDA，并初始化相关张量。

        参数:
            gaus (util_gau.GaussianData 或 dict): 高斯点云数据或其字典。
        """
        self.need_rerender = True
        if type(gaus) is dict:
            gau_xyz = []
            gau_rot = []
            gau_s = []
            gau_a = []
            gau_c = []
            for gaus_item in gaus.values():
                gau_xyz.append(gaus_item.xyz)
                gau_rot.append(gaus_item.rot)
                gau_s.append(gaus_item.scale)
                gau_a.append(gaus_item.opacity)
                gau_c.append(gaus_item.sh)
            self.gau_env_idx = gau_xyz[0].shape[0]
            gau_xyz = np.concatenate(gau_xyz, axis=0)
            gau_rot = np.concatenate(gau_rot, axis=0)
            gau_s = np.concatenate(gau_s, axis=0)
            gau_a = np.concatenate(gau_a, axis=0)
            gau_c = np.concatenate(gau_c, axis=0)
            gaus_all = util_gau.GaussianData(gau_xyz, gau_rot, gau_s, gau_a, gau_c)
            self.gaussians = gaus_cuda_from_cpu(gaus_all)
        else:
            self.gaussians = gaus_cuda_from_cpu(gaus)
        self.raster_settings.sh_degree = int(np.round(np.sqrt(self.gaussians.sh_dim))) - 1

        num_points = self.gaussians.xyz.shape[0]

        self.gau_ori_xyz_all_cu = torch.zeros(num_points, 3).cuda().requires_grad_(False)
        self.gau_ori_xyz_all_cu[..., :] = torch.from_numpy(gau_xyz).cuda().requires_grad_(False)
        self.gau_ori_rot_all_cu = torch.zeros(num_points, 4).cuda().requires_grad_(False)
        self.gau_ori_rot_all_cu[..., :] = torch.from_numpy(gau_rot).cuda().requires_grad_(False)

        self.gau_xyz_all_cu = torch.zeros(num_points, 3).cuda().requires_grad_(False)
        self.gau_rot_all_cu = torch.zeros(num_points, 4).cuda().requires_grad_(False)

    def set_scale_modifier(self, modifier):
        """
        设置高斯点云的缩放因子。

        参数:
            modifier (float): 缩放因子。
        """
        self.need_rerender = True
        self.raster_settings.scale_modifier = float(modifier)

    def set_render_reso(self, w, h):
        """
        设置渲染输出的分辨率。

        参数:
            w (int): 新的宽度。
            h (int): 新的高度。
        """
        self.need_rerender = True
        self.raster_settings.image_height = int(h)
        self.raster_settings.image_width = int(w)

    def update_camera_pose(self, camera: util_gau.Camera):
        """
        根据相机对象更新视图矩阵和投影矩阵。

        参数:
            camera (util_gau.Camera): 相机对象。
        """
        self.need_rerender = True
        view_matrix = camera.get_view_matrix()
        view_matrix[[0,2], :] *= -1
        
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings.viewmatrix = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings.campos = torch.tensor(camera.position).float().cuda()
        self.raster_settings.projmatrix = torch.tensor(proj.T).float().cuda()

    def update_camera_pose_from_topic(self, camera: util_gau.Camera, rmat, trans):
        """
        根据外部输入的旋转矩阵和平移向量更新相机位姿。

        参数:
            camera (util_gau.Camera): 相机对象。
            rmat (np.ndarray): 旋转矩阵 (3, 3)。
            trans (np.ndarray): 平移向量 (3,)。
        """
        self.need_rerender = True

        camera.position = np.array(trans).astype(np.float32)
        camera.target = camera.position - (1. * rmat[:3,2]).astype(np.float32)

        Tmat = np.eye(4)
        Tmat[:3,:3] = rmat
        Tmat[:3,3] = trans
        Tmat[0:3, [1,2]] *= -1
        transpose = np.array([[-1.0,  0.0,  0.0,  0.0],
                              [ 0.0, -1.0,  0.0,  0.0],
                              [ 0.0,  0.0,  1.0,  0.0],
                              [ 0.0,  0.0,  0.0,  1.0]])
        view_matrix = transpose @ np.linalg.inv(Tmat)

        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings.projmatrix = torch.tensor(proj.T).float().cuda()
        self.raster_settings.viewmatrix = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings.campos = torch.tensor(camera.position).float().cuda()

    def update_camera_intrin(self, camera: util_gau.Camera):
        """
        更新相机的内参（视场角等）。

        参数:
            camera (util_gau.Camera): 相机对象。
        """
        hfovx, hfovy, focal = camera.get_htanfovxy_focal()
        self.raster_settings.tanfovx = hfovx
        self.raster_settings.tanfovy = hfovy

    def draw(self, render_depth=False):
        """
        执行高斯点云的渲染，返回RGB或深度图。

        参数:
            render_depth (bool): 是否渲染深度图。

        返回:
            np.ndarray: 渲染的RGB图像或深度图。
        """
        if not self.need_rerender:
            if render_depth:
                return self.render_depth_img
            else:
                return self.render_rgb_img

        self.need_rerender = False

        rasterizer = GausssianRasterizer_3d(raster_settings=self.raster_settings)

        with torch.no_grad():
            color_img, radii, depth_img, _alpha = rasterizer(
                means3D = self.gaussians.xyz,
                means2D = None,
                shs = self.gaussians.sh,
                colors_precomp = None,
                opacities = self.gaussians.opacity,
                scales = self.gaussians.scale,
                rotations = self.gaussians.rot,
                cov3D_precomp = None
            )

            self.render_depth_img = depth_img.permute(1, 2, 0).contiguous().cpu().numpy()
            self.render_rgb_img = (255. * torch.clamp(color_img, 0.0, 1.0)).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()

        if render_depth:
            return self.render_depth_img
        else:
            return self.render_rgb_img