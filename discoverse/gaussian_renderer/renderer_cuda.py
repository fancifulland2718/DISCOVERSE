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
    �洢��˹���Ƶ�CUDA�������ݽṹ��

    ����:
        xyz (torch.Tensor): ��˹�����ά���� (N, 3)��
        rot (torch.Tensor): ��˹�����ת��Ԫ�� (N, 4)��
        scale (torch.Tensor): ��˹������Ų��� (N, 3)��
        opacity (torch.Tensor): ��˹��Ĳ�͸���� (N, 1)��
        sh (torch.Tensor): ��гϵ�� (N, D, 3)��
    """
    xyz: torch.Tensor
    rot: torch.Tensor
    scale: torch.Tensor
    opacity: torch.Tensor
    sh: torch.Tensor
    
    def __len__(self):
        """
        ���ظ�˹���������
        """
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        """
        ������гϵ����ά�ȡ�
        """
        return self.sh.shape[-2]

@dataclass
class GaussianRasterizationSettingsStorage:
    """
    �洢��˹���ƹ�դ������Ⱦ���á�

    ����:
        image_height (int): ���ͼ��߶ȡ�
        image_width (int): ���ͼ���ȡ�
        tanfovx (float): ˮƽ�ӳ��ǵ�����ֵ��
        tanfovy (float): ��ֱ�ӳ��ǵ�����ֵ��
        bg (torch.Tensor): ������ɫ (3,)��
        scale_modifier (float): �������ӡ�
        viewmatrix (torch.Tensor): ��ͼ���� (4, 4)��
        projmatrix (torch.Tensor): ͶӰ���� (4, 4)��
        sh_degree (int): ��гչ���Ľ�����
        campos (torch.Tensor): ���λ�� (3,)��
        prefiltered (bool): �Ƿ�Ԥ�˲���
        debug (bool): �Ƿ�������ģʽ��
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
    ��CPU�ϵĸ�˹��������ת��ΪCUDA������

    ����:
        gau (util_gau.GaussianData): CPU�ϵĸ�˹�������ݡ�

    ����:
        GaussianDataCUDA: ����CUDA�����ĸ�˹�������ݡ�
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
    ����CUDA�ĸ�˹������Ⱦ������������˹�����ݡ������������Ⱦ���̡�
    """
    def __init__(self, w, h):
        """
        ��ʼ����Ⱦ����������Ⱦ�ֱ��ʺ�Ĭ�ϲ�����

        ����:
            w (int): ��Ⱦ��ȡ�
            h (int): ��Ⱦ�߶ȡ�
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
        ���¸�˹�������ݵ�CUDA������ʼ�����������

        ����:
            gaus (util_gau.GaussianData �� dict): ��˹�������ݻ����ֵ䡣
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
        ���ø�˹���Ƶ��������ӡ�

        ����:
            modifier (float): �������ӡ�
        """
        self.need_rerender = True
        self.raster_settings.scale_modifier = float(modifier)

    def set_render_reso(self, w, h):
        """
        ������Ⱦ����ķֱ��ʡ�

        ����:
            w (int): �µĿ�ȡ�
            h (int): �µĸ߶ȡ�
        """
        self.need_rerender = True
        self.raster_settings.image_height = int(h)
        self.raster_settings.image_width = int(w)

    def update_camera_pose(self, camera: util_gau.Camera):
        """
        ����������������ͼ�����ͶӰ����

        ����:
            camera (util_gau.Camera): �������
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
        �����ⲿ�������ת�����ƽ�������������λ�ˡ�

        ����:
            camera (util_gau.Camera): �������
            rmat (np.ndarray): ��ת���� (3, 3)��
            trans (np.ndarray): ƽ������ (3,)��
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
        ����������ڲΣ��ӳ��ǵȣ���

        ����:
            camera (util_gau.Camera): �������
        """
        hfovx, hfovy, focal = camera.get_htanfovxy_focal()
        self.raster_settings.tanfovx = hfovx
        self.raster_settings.tanfovy = hfovy

    def draw(self, render_depth=False):
        """
        ִ�и�˹���Ƶ���Ⱦ������RGB�����ͼ��

        ����:
            render_depth (bool): �Ƿ���Ⱦ���ͼ��

        ����:
            np.ndarray: ��Ⱦ��RGBͼ������ͼ��
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