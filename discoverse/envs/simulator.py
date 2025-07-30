import os
import sys
import time
import traceback
from abc import abstractmethod
import ctypes

import cv2
import glfw
from PIL import Image
import OpenGL.GL as gl
from OpenGL.GL import shaders

import mujoco
import random
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse import DISCOVERSE_ASSETS_DIR
from discoverse.utils import BaseConfig

if sys.platform == "linux":
    try:
        from discoverse.gaussian_renderer import GSRenderer
        from discoverse.gaussian_renderer.util_gau import multiple_quaternion_vector3d, multiple_quaternions
        DISCOVERSE_GAUSSIAN_RENDERER = True

    except ImportError:
        traceback.print_exc()
        print("Warning: gaussian_splatting renderer not found. Please install the required packages to use it.")
        DISCOVERSE_GAUSSIAN_RENDERER = False
else:
    DISCOVERSE_GAUSSIAN_RENDERER = False

def setRenderOptions(options):
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True
    options.frame = mujoco.mjtFrame.mjFRAME_BODY.value
    pass

class SimulatorBase:
    running = True
    obs = None
    img_rgb_obs_s = {}
    img_depth_obs_s = {}
    free_body_qpos_ids = {}

    cam_id = -1  # -1表示自由视角
    last_cam_id = -1
    render_cnt = 0
    camera_names = []
    camera_pose_changed = False
    camera_rmat = np.array([
        [ 0,  0, -1],
        [-1,  0,  0],
        [ 0,  1,  0],
    ])

    use_default_window_size = False
    mouse_pressed = {
        'left': False,
        'right': False,
        'middle': False
    }
    mouse_pos = {
        'x': 0,
        'y': 0
    }

    options = mujoco.MjvOption()

    def __init__(self, config:BaseConfig):
        self.config = config
        
        # 初始化OpenGL资源变量
        self.VAO = 0
        self.VBO = 0
        self.EBO = 0
        self.texture = 0
        self.shader_program = 0

        if self.config.mjcf_file_path.startswith("/"):
            self.mjcf_file = self.config.mjcf_file_path
        elif os.path.exists(self.config.mjcf_file_path):
            self.mjcf_file = self.config.mjcf_file_path
        else:
            self.mjcf_file = os.path.join(DISCOVERSE_ASSETS_DIR, self.config.mjcf_file_path)
        if os.path.exists(self.mjcf_file):
            print("mjcf found: {}".format(self.mjcf_file))
        else:
            print("\033[0;31;40mFailed to load mjcf: {}\033[0m".format(self.mjcf_file))
            raise FileNotFoundError("Failed to load mjcf: {}".format(self.mjcf_file))
        self.load_mjcf()
        self.decimation = self.config.decimation
        self.delta_t = self.mj_model.opt.timestep * self.decimation
        self.render_fps = self.config.render_set["fps"]

        if self.config.enable_render:
            self.free_camera = mujoco.MjvCamera()
            self.free_camera.fixedcamid = -1
            self.free_camera.type = mujoco._enums.mjtCamera.mjCAMERA_FREE
            mujoco.mjv_defaultFreeCamera(self.mj_model, self.free_camera)

            self.config.use_gaussian_renderer = self.config.use_gaussian_renderer and DISCOVERSE_GAUSSIAN_RENDERER
            if self.config.use_gaussian_renderer:
                self.gs_renderer = GSRenderer(self.config.gs_model_dict, self.config.render_set["width"], self.config.render_set["height"])
                self.last_cam_id = self.cam_id
                self.show_gaussian_img = True
                if self.cam_id == -1:
                    self.gs_renderer.set_camera_fovy(self.mj_model.vis.global_.fovy * np.pi / 180.)
                else:
                    self.gs_renderer.set_camera_fovy(self.mj_model.cam_fovy[self.cam_id] * np.pi / 180.0)

        self.window = None
        self.glfw_initialized = False
        
        if not hasattr(self.config.render_set, "window_title"):
            self.config.render_set["window_title"] = "DISCOVERSE"
        
        if self.config.enable_render and not self.config.headless:
            try:
                if not glfw.init():
                    raise RuntimeError("无法初始化GLFW")
                self.glfw_initialized = True
                
                # 设置OpenGL版本
                glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
                glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
                glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
                glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
                glfw.window_hint(glfw.VISIBLE, True)

                # 如果设置了use_default_window_size，禁用窗口最大化功能
                if self.use_default_window_size:
                    glfw.window_hint(glfw.MAXIMIZED, False)
                    glfw.window_hint(glfw.DECORATED, True)
                    glfw.window_hint(glfw.RESIZABLE, True)
                    print("已禁用窗口最大化功能，但允许调整窗口大小")

                # 创建窗口
                self.window = glfw.create_window(
                    self.config.render_set["width"],
                    self.config.render_set["height"],
                    self.config.render_set.get("window_title", "DISCOVERSE"),
                    None, None
                )
                
                if not self.window:
                    glfw.terminate()
                    raise RuntimeError("无法创建GLFW窗口")
                
                glfw.make_context_current(self.window)
                glfw.swap_interval(1)

                # 设置窗口大小限制
                glfw.set_window_size_limits(self.window, 320, 240, 
                                          self.mj_model.vis.global_.offwidth, 
                                          self.mj_model.vis.global_.offheight)

                # 基本OpenGL设置
                gl.glClearColor(0.0, 0.0, 0.0, 1.0)
                
                # 初始化OpenGL资源
                self._init_modern_gl_resources()
                
                # 设置回调
                glfw.set_key_callback(self.window, self.on_key)
                glfw.set_cursor_pos_callback(self.window, self.on_mouse_move)
                glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
                glfw.set_scroll_callback(self.window, self.on_mouse_scroll)
                
                # 如果设置了use_default_window_size，添加窗口大小变化回调
                if self.use_default_window_size:
                    glfw.set_window_maximize_callback(self.window, self.maximize_callback)

                # 跨平台显示缩放支持
                if sys.platform == "darwin":  # macOS
                    try:
                        import AppKit
                        def get_screen_scale(screen_id):
                            screens = AppKit.NSScreen.screens()
                            if len(screens) > screen_id:
                                return screens[screen_id].backingScaleFactor()
                            else:
                                return 2.0  # 默认retina缩放
                        self.screen_scale = get_screen_scale(0)
                    except ImportError:
                        print("Warning: pyobjc not available for retina display support on macOS")
                        print("Using default scale. Install with: pip install pyobjc")
                        self.screen_scale = 2.0  # 默认retina缩放
                else:
                    self.screen_scale = 1.0

                # 注册清理函数
                import atexit
                atexit.register(self._cleanup_before_exit)

            except Exception as e:
                print(f"GLFW初始化失败: {e}")
                if self.glfw_initialized:
                    glfw.terminate()
                self.config.headless = True
                self.window = None

        self.last_render_time = time.time()
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def maximize_callback(self, window, maximized):
        """窗口最大化回调，用于use_default_window_size功能"""
        if self.use_default_window_size and maximized:
            glfw.restore_window(window)

    def _init_modern_gl_resources(self):
        """初始化OpenGL资源用于简单图像显示"""
        try:
            # 检查OpenGL上下文
            if not glfw.get_current_context():
                raise RuntimeError("OpenGL上下文未激活")
            
            # 创建简单的全屏四边形顶点数据
            vertices = np.array([
                -1.0, -1.0,  0.0, 0.0,  # 左下
                 1.0, -1.0,  1.0, 0.0,  # 右下
                 1.0,  1.0,  1.0, 1.0,  # 右上
                -1.0,  1.0,  0.0, 1.0   # 左上
            ], dtype=np.float32)

            indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

            # 创建VAO, VBO, EBO
            self.VAO = gl.glGenVertexArrays(1)
            self.VBO = gl.glGenBuffers(1)
            self.EBO = gl.glGenBuffers(1)

            gl.glBindVertexArray(self.VAO)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.VBO)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.EBO)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

            # 设置顶点属性
            gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(0))
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(8))
            gl.glEnableVertexAttribArray(1)

            # 简化的着色器
            vertex_shader_source = """
            #version 330 core
            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 aTexCoord;
            out vec2 TexCoord;
            void main() {
                gl_Position = vec4(aPos, 0.0, 1.0);
                TexCoord = aTexCoord;
            }
            """

            fragment_shader_source = """
            #version 330 core
            out vec4 FragColor;
            in vec2 TexCoord;
            uniform sampler2D ourTexture;
            void main() {
                FragColor = texture(ourTexture, TexCoord);
            }
            """

            # 编译着色器程序
            vertex_shader = shaders.compileShader(vertex_shader_source, gl.GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_shader_source, gl.GL_FRAGMENT_SHADER)
            self.shader_program = shaders.compileProgram(vertex_shader, fragment_shader)

            # 创建纹理
            self.texture = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

            # 解绑
            gl.glBindVertexArray(0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                
        except Exception as e:
            print(f"初始化OpenGL资源失败: {e}")
            self._cleanup_opengl_resources()
            raise

    def _render_image_modern_gl(self, img_vis):
        """使用OpenGL渲染图像到窗口"""
        if img_vis is None or not hasattr(self, 'VAO') or not self.VAO:
            return
            
        try:
            # 翻转图像（OpenGL坐标系）
            img_vis = img_vis[::-1]
            img_vis = np.ascontiguousarray(img_vis)
            height, width = img_vis.shape[:2]
            
            # 更新纹理
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_vis.tobytes())
            
            # 渲染
            gl.glUseProgram(self.shader_program)
            gl.glBindVertexArray(self.VAO)
            gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)
                
        except Exception as e:
            print(f"渲染错误: {e}")

    def object_pose(self, body_name):
        """获取物体的位姿（位置xyz和朝向wxyz）"""
        try:
            qid = self.mj_model.jnt_qposadr[self.free_body_qpos_ids[body_name]]
            return self.mj_data.qpos[qid:qid+7][...]
        except KeyError:
            raise KeyError(f"Body name '{body_name}' not found in free_body_qpos_ids. Available bodies: {list(self.free_body_qpos_ids.keys())}")
    
    def get_joint_position(self, joint_name):
        return self.mj_data.qpos[self.mj_model.joint(joint_name).qposadr]
    
    def set_joint_position(self, joint_name, value):
        self.mj_data.qpos[self.mj_model.joint(joint_name).qposadr] = value

    def load_mjcf(self):
        if self.mjcf_file.endswith(".xml"):
            self.mj_model = mujoco.MjModel.from_xml_path(self.mjcf_file)
        elif self.mjcf_file.endswith(".mjb"):
            self.mj_model = mujoco.MjModel.from_binary_path(self.mjcf_file)
        self.mj_model.opt.timestep = self.config.timestep
        # self.mj_model.vis.quality.shadowsize = 4096 * 8
        self.mj_data = mujoco.MjData(self.mj_model)
        if self.config.enable_render:
            for i in range(self.mj_model.ncam):
                self.camera_names.append(self.mj_model.camera(i).name)

            if type(self.config.obs_rgb_cam_id) is int:
                assert -2 < self.config.obs_rgb_cam_id < len(self.camera_names), "Invalid obs_rgb_cam_id {}".format(self.config.obs_rgb_cam_id)
                tmp_id = self.config.obs_rgb_cam_id
                self.config.obs_rgb_cam_id = [tmp_id]
            elif type(self.config.obs_rgb_cam_id) is list:
                for cam_id in self.config.obs_rgb_cam_id:
                    assert -2 < cam_id < len(self.camera_names), "Invalid obs_rgb_cam_id {}".format(cam_id)
            elif self.config.obs_rgb_cam_id is None:
                self.config.obs_rgb_cam_id = []
            
            if type(self.config.obs_depth_cam_id) is int:
                assert -2 < self.config.obs_depth_cam_id < len(self.camera_names), "Invalid obs_depth_cam_id {}".format(self.config.obs_depth_cam_id)
            elif type(self.config.obs_depth_cam_id) is list:
                for cam_id in self.config.obs_depth_cam_id:
                    assert -2 < cam_id < len(self.camera_names), "Invalid obs_depth_cam_id {}".format(cam_id)
            elif self.config.obs_depth_cam_id is None:
                self.config.obs_depth_cam_id = []
        
            try:
                # 跨平台屏幕信息检测
                screen_width, screen_height = 1920, 1080  # 默认值
                
                if sys.platform == "darwin":  # macOS
                    try:
                        import AppKit
                        screens = AppKit.NSScreen.screens()
                        if screens:
                            main_screen = screens[0]
                            frame = main_screen.frame()
                            screen_width = int(frame.size.width)
                            screen_height = int(frame.size.height)
                    except ImportError:
                        print("Warning: pyobjc not available for screen detection on macOS")
                        self.use_default_window_size = True
                elif sys.platform.startswith("linux"):  # Ubuntu/Linux
                    try:
                        import screeninfo
                        monitors = screeninfo.get_monitors()
                        for m in monitors:
                            if m.is_primary:
                                screen_width, screen_height = m.width, m.height
                                break
                    except ImportError:
                        print("Warning: screeninfo not available for screen detection on Linux")
                        self.use_default_window_size = True
                    except Exception as e:
                        print(f"screeninfo error: {e}, using default screen size")
                        self.use_default_window_size = True
                else:
                    # Windows或其他系统
                    self.use_default_window_size = True
                    
            except Exception as e:
                screen_width, screen_height = 1920, 1080
                print(f"Screen detection error: {e}, using default screen size: {screen_width}x{screen_height}")
                self.use_default_window_size = True

            self.mj_model.vis.global_.offwidth = max(self.mj_model.vis.global_.offwidth, screen_width)
            self.mj_model.vis.global_.offheight = max(self.mj_model.vis.global_.offheight, screen_height)
            self.renderer = mujoco.Renderer(self.mj_model, self.config.render_set["height"], self.config.render_set["width"])

        for i in range(self.mj_model.nbody):
            if len(self.mj_model.body(i).name) and self.mj_model.body(i).dofnum == 6:
                jq_id = np.where(self.mj_model.jnt_bodyid == self.mj_model.body(i).id)[0]
                if jq_id.size:
                    self.free_body_qpos_ids[self.mj_model.body(i).name] = int(jq_id[0])

        self.post_load_mjcf()

    def post_load_mjcf(self):
        pass

    def update_renderer_window_size(self, width, height):
        self.renderer._width = width
        self.renderer._height = height
        self.renderer._rect.width = width
        self.renderer._rect.height = height
        if self.config.use_gaussian_renderer and self.show_gaussian_img:
            self.gs_renderer.set_camera_resolution(height, width)

    def get_random_texture(self):
        TEXTURE_1K_PATH = os.getenv("TEXTURE_1K_PATH", os.path.join(DISCOVERSE_ASSETS_DIR, "textures_1k"))
        if not TEXTURE_1K_PATH is None and os.path.exists(TEXTURE_1K_PATH):
            return Image.open(os.path.join(TEXTURE_1K_PATH, random.choice(os.listdir(TEXTURE_1K_PATH))))
        else:
            # raise ValueError("TEXTURE_1K_PATH not found")
            print("Warning: TEXTURE_1K_PATH not found! Please set the TEXTURE_1K_PATH environment variable to the path of the textures_1k directory.")
            return Image.fromarray(np.random.randint(0, 255, (768, 768, 3), dtype=np.uint8))

    def update_texture(self, texture_name, mtl_img_pil, no_render=False):
        """更新纹理"""
        if not hasattr(self, 'renderer') or self.renderer is None:
            print(f"Renderer not initialized, cannot update texture: {texture_name}")
            return False

        try:
            tex_id = self.renderer.model.tex(texture_name).id
        except Exception as e:
            print(f"Texture '{texture_name}' not found: {e}")
            return False
        
        if not no_render:
            self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            self.renderer.render()
        
        tex_bind_id = self.renderer._mjr_context.texture[tex_id]
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_bind_id)
        
        try:
            width = gl.glGetTexLevelParameteriv(gl.GL_TEXTURE_2D, 0, gl.GL_TEXTURE_WIDTH)
            height = gl.glGetTexLevelParameteriv(gl.GL_TEXTURE_2D, 0, gl.GL_TEXTURE_HEIGHT)
        except Exception as e:
            print(f"Error getting texture dimensions: {e}")
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            return False

        try:
            if mtl_img_pil.mode != 'RGB':
                mtl_img_pil = mtl_img_pil.convert('RGB')

            if mtl_img_pil.size != (width, height):
                mtl_img_pil = mtl_img_pil.resize((width, height), Image.Resampling.LANCZOS)
            
            mtl_img = np.array(mtl_img_pil)
            mtl_img = np.flipud(mtl_img)
            mtl_img = np.ascontiguousarray(mtl_img, dtype=np.uint8)

            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)

            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, width, height, 
                              gl.GL_RGB, gl.GL_UNSIGNED_BYTE, mtl_img.tobytes())
            
        except Exception as e:
            print(f"Error processing image for texture '{texture_name}': {e}")
            return False
        finally:
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            
        return True

    def render(self):
        self.render_cnt += 1

        self.update_renderer_window_size(self.config.render_set["width"], self.config.render_set["height"])
        if self.config.use_gaussian_renderer and self.show_gaussian_img:
            self.update_gs_scene()
        
        depth_rendering = self.renderer._depth_rendering
        self.renderer.disable_depth_rendering()
        for id in self.config.obs_rgb_cam_id:
            img = self.getRgbImg(id)
            self.img_rgb_obs_s[id] = img
        
        self.renderer.enable_depth_rendering()
        for id in self.config.obs_depth_cam_id:
            img = self.getDepthImg(id)
            self.img_depth_obs_s[id] = img
        self.renderer._depth_rendering = depth_rendering
        
        if not self.config.headless and self.window is not None:
            current_width_s_, current_height_s_ = glfw.get_framebuffer_size(self.window)
            current_width, current_height = int(current_width_s_/self.screen_scale), int(current_height_s_/self.screen_scale)
            if current_height == self.config.render_set["height"] and current_width == self.config.render_set["width"]:
                if not self.renderer._depth_rendering:
                    if self.cam_id in self.config.obs_rgb_cam_id:
                        img_vis = self.img_rgb_obs_s[self.cam_id]
                    else:
                        img_rgb = self.getRgbImg(self.cam_id)
                        img_vis = img_rgb
                else:
                    if self.cam_id in self.config.obs_depth_cam_id:
                        img_depth = self.img_depth_obs_s[self.cam_id]
                    else:
                        img_depth = self.getDepthImg(self.cam_id)
                    
                    if img_depth is not None:
                        img_vis = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=255./self.config.max_render_depth), cv2.COLORMAP_JET)
                    else:
                        img_vis = None
            else:
                self.update_renderer_window_size(current_width, current_height)
                if not self.renderer._depth_rendering:
                    img_vis = self.getRgbImg(self.cam_id)
                else:
                    img_depth = self.getDepthImg(self.cam_id)
                    img_vis = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=255./self.config.max_render_depth), cv2.COLORMAP_JET)

            try:
                if glfw.window_should_close(self.window):
                    self.running = False
                    return
                    
                glfw.make_context_current(self.window)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)

                if img_vis is not None:
                    self._render_image_modern_gl(img_vis)
                
                glfw.swap_buffers(self.window)
                glfw.poll_events()
                
                if self.config.sync:
                    current_time = time.time()
                    wait_time = max(1.0/self.render_fps - (current_time - self.last_render_time), 0)
                    if wait_time > 0:
                        time.sleep(wait_time)
                    self.last_render_time = time.time()
                    
            except Exception as e:
                print(f"渲染错误: {e}")

    def getRgbImg(self, cam_id):
        if self.config.use_gaussian_renderer and self.show_gaussian_img:
            if cam_id == -1:
                self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
                self.gs_renderer.set_camera_fovy(self.mj_model.vis.global_.fovy * np.pi / 180.0)
            if self.last_cam_id != cam_id and cam_id > -1:
                self.gs_renderer.set_camera_fovy(self.mj_model.cam_fovy[cam_id] * np.pi / 180.0)
            self.last_cam_id = cam_id
            trans, quat_wxyz = self.getCameraPose(cam_id)
            self.gs_renderer.set_camera_pose(trans, quat_wxyz[[1,2,3,0]])
            return self.gs_renderer.render()
        else:
            if cam_id == -1:
                self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            elif cam_id > -1:
                self.renderer.update_scene(self.mj_data, self.camera_names[cam_id], self.options)
            else:
                return None
            rgb_img = self.renderer.render()
            return rgb_img

    def getDepthImg(self, cam_id):
        if self.config.use_gaussian_renderer and self.show_gaussian_img:
            if cam_id == -1:
                self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            if self.last_cam_id != cam_id:
                if cam_id == -1:
                    self.gs_renderer.set_camera_fovy(np.pi * 0.25)
                elif cam_id > -1:
                    self.gs_renderer.set_camera_fovy(self.mj_model.cam_fovy[cam_id] * np.pi / 180.0)
                else:
                    return None
            self.last_cam_id = cam_id
            trans, quat_wxyz = self.getCameraPose(cam_id)
            self.gs_renderer.set_camera_pose(trans, quat_wxyz[[1,2,3,0]])
            return self.gs_renderer.render(render_depth=True)
        else:
            if cam_id == -1:
                self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            elif cam_id > -1:
                self.renderer.update_scene(self.mj_data, self.camera_names[cam_id], self.options)
            else:
                return None
            depth_img = self.renderer.render()
            return depth_img

    def getPointCloud(self, cam_id, N_gap=5):
        """ please call after get_observation """

        assert (cam_id in self.config.obs_rgb_cam_id) and (cam_id in self.config.obs_depth_cam_id), "Invalid cam_id"

        # 相机内参预计算
        fovy = (self.mj_model.vis.global_.fovy if cam_id == -1 else self.mj_model.cam_fovy[cam_id]) * np.pi / 180.0
        height, width = self.config.render_set["height"], self.config.render_set["width"]
        fx = width / (2 * np.tan(fovy * width / (2 * height)))  # 合并计算
        fy = height / (2 * np.tan(fovy / 2))
        cx, cy = width / 2, height / 2
        inv_fx, inv_fy = 1.0 / fx, 1.0 / fy

        rgb = self.obs["rgb"][cam_id][::N_gap, ::N_gap]
        depth = self.obs["depth"][cam_id][::N_gap, ::N_gap]

        rows, cols = depth.shape
        u = (np.arange(cols) * N_gap).astype(np.float32)
        v = (np.arange(rows) * N_gap).astype(np.float32)
        u, v = np.meshgrid(u, v)
        u_flat = u.ravel()
        v_flat = v.ravel()
        depth_flat = depth.ravel()

        valid = depth_flat > 0
        u_valid = u_flat[valid]
        v_valid = v_flat[valid]
        Z_valid = depth_flat[valid]

        points = np.empty((len(Z_valid), 3), dtype=np.float32)
        points[:, 0] = (u_valid - cx) * Z_valid * inv_fx
        points[:, 1] = (v_valid - cy) * Z_valid * inv_fy
        points[:, 2] = Z_valid

        colors_rgb = rgb.reshape(-1, 3)[valid].astype(np.float32) / 255.0

        return points, colors_rgb

    def on_mouse_move(self, window, xpos, ypos):
        if self.cam_id == -1:
            dx = xpos - self.mouse_pos['x']
            dy = ypos - self.mouse_pos['y']
            height = self.config.render_set["height"]
            
            action = None
            if self.mouse_pressed['left']:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
            elif self.mouse_pressed['right']:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_V
            elif self.mouse_pressed['middle']:
                action = mujoco.mjtMouse.mjMOUSE_ZOOM

            if action is not None:
                self.camera_pose_changed = True
                mujoco.mjv_moveCamera(self.mj_model,  action,  dx/height,  dy/height, self.renderer.scene, self.free_camera)

        self.mouse_pos['x'] = xpos
        self.mouse_pos['y'] = ypos

    def on_mouse_button(self, window, button, action, mods):
        is_pressed = action == glfw.PRESS
        
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_pressed['left'] = is_pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse_pressed['right'] = is_pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.mouse_pressed['middle'] = is_pressed

    def on_mouse_scroll(self, window, xoffset, yoffset):
        self.free_camera.distance -= yoffset * 0.1
        if self.free_camera.distance < 0.1:
            self.free_camera.distance = 0.1

    def on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            is_ctrl_pressed = (mods & glfw.MOD_CONTROL)
            
            if is_ctrl_pressed:
                if key == glfw.KEY_G:  # Ctrl + G
                    if self.config.use_gaussian_renderer:
                        self.show_gaussian_img = not self.show_gaussian_img
                        self.gs_renderer.renderer.need_rerender = True
                elif key == glfw.KEY_D:  # Ctrl + D
                    if self.config.use_gaussian_renderer:
                        self.gs_renderer.renderer.need_rerender = True
                    if self.renderer._depth_rendering:
                        self.renderer.disable_depth_rendering()
                    else:
                        self.renderer.enable_depth_rendering()
            else:
                if key == glfw.KEY_H:  # 'h': 显示帮助
                    self.printHelp()
                elif key == glfw.KEY_P:  # 'p': 打印信息
                    self.printMessage()
                elif key == glfw.KEY_R:  # 'r': 重置状态
                    self.reset()
                elif key == glfw.KEY_ESCAPE:  # ESC: 切换到自由视角
                    self.cam_id = -1
                    self.camera_pose_changed = True
                elif key == glfw.KEY_RIGHT_BRACKET:  # ']': 下一个相机
                    if self.mj_model.ncam:
                        self.cam_id += 1
                        self.cam_id = self.cam_id % self.mj_model.ncam
                elif key == glfw.KEY_LEFT_BRACKET:  # '[': 上一个相机
                    if self.mj_model.ncam:
                        self.cam_id += self.mj_model.ncam - 1
                        self.cam_id = self.cam_id % self.mj_model.ncam

    def printHelp(self):
        """打印帮助信息"""
        print("\n=== 键盘控制说明 ===")
        print("H: 显示此帮助信息")
        print("P: 打印当前状态信息")
        print("R: 重置模拟器状态")
        print("G: 切换高斯渲染（如果可用）")
        print("D: 切换深度渲染")
        print("Ctrl+G: 组合键切换高斯模式")
        print("Ctrl+D: 组合键切换深度图模式")
        print("ESC: 切换到自由视角")
        print("[: 切换到上一个相机")
        print("]: 切换到下一个相机")
        print("\n=== 鼠标控制说明 ===")
        print("左键拖动: 旋转视角")
        print("右键拖动: 平移视角")
        print("中键拖动: 缩放视角")
        print("================\n")

    def printMessage(self):
        """打印当前状态信息"""
        print("\n=== 当前状态 ===")
        print(f"当前相机ID: {self.cam_id}")
        if self.cam_id >= 0:
            print(f"相机名称: {self.camera_names[self.cam_id]}")
        print(f"高斯渲染: {'开启' if self.show_gaussian_img else '关闭'}")
        print(f"深度渲染: {'开启' if self.renderer._depth_rendering else '关闭'}")
        print("==============\n")

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.camera_pose_changed = True

    def update_gs_scene(self):
        for name in self.config.obj_list + self.config.rb_link_list:
            trans, quat_wxyz = self.getObjPose(name)
            self.gs_renderer.set_obj_pose(name, trans, quat_wxyz)

        if self.gs_renderer.update_gauss_data:
            self.gs_renderer.update_gauss_data = False
            self.gs_renderer.renderer.need_rerender = True
            self.gs_renderer.renderer.gaussians.xyz[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternion_vector3d(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]) + self.gs_renderer.renderer.gau_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]
            self.gs_renderer.renderer.gaussians.rot[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternions(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:])

    def getObjPose(self, name):
        try:
            position = self.mj_data.body(name).xpos
            quat = self.mj_data.body(name).xquat
            return position, quat
        except KeyError:
            try:
                position = self.mj_data.geom(name).xpos
                quat = Rotation.from_matrix(self.mj_data.geom(name).xmat.reshape((3,3))).as_quat()[[3,0,1,2]]
                return position, quat
            except KeyError:
                print("Invalid object name: {}".format(name))
                return None, None
    
    def getCameraPose(self, cam_id):
        if cam_id == -1:
            rotation_matrix = self.camera_rmat @ Rotation.from_euler('xyz', [self.free_camera.elevation * np.pi / 180.0, self.free_camera.azimuth * np.pi / 180.0, 0.0]).as_matrix()
            camera_position = self.free_camera.lookat + self.free_camera.distance * rotation_matrix[:3,2]
        else:
            rotation_matrix = np.array(self.mj_data.camera(self.camera_names[cam_id]).xmat).reshape((3,3))
            camera_position = self.mj_data.camera(self.camera_names[cam_id]).xpos

        return camera_position, Rotation.from_matrix(rotation_matrix).as_quat()[[3,0,1,2]]

    def _cleanup_opengl_resources(self):
        """清理OpenGL资源"""
        try:
            if hasattr(self, 'window') and self.window and glfw.get_current_context() == self.window:
                if hasattr(self, 'VAO') and self.VAO:
                    gl.glDeleteVertexArrays(1, [self.VAO])
                    self.VAO = 0
                if hasattr(self, 'VBO') and self.VBO:
                    gl.glDeleteBuffers(1, [self.VBO])
                    self.VBO = 0
                if hasattr(self, 'EBO') and self.EBO:
                    gl.glDeleteBuffers(1, [self.EBO])
                    self.EBO = 0
                if hasattr(self, 'texture') and self.texture:
                    gl.glDeleteTextures(1, [self.texture])
                    self.texture = 0
                if hasattr(self, 'shader_program') and self.shader_program:
                    gl.glDeleteProgram(self.shader_program)
                    self.shader_program = 0
        except Exception:
            pass

    def _cleanup_before_exit(self):
        """程序退出前的清理工作"""
        try:
            self._cleanup_opengl_resources()
            
            if hasattr(self, 'renderer'):
                try:
                    del self.renderer
                except Exception:
                    pass

            if hasattr(self, 'window') and self.window is not None:
                try:
                    glfw.destroy_window(self.window)
                except Exception:
                    pass
                self.window = None
            
            if hasattr(self, 'glfw_initialized') and self.glfw_initialized:
                try:
                    glfw.terminate()
                except Exception:
                    pass
                self.glfw_initialized = False
            
        except Exception:
            pass

    # ------------------------------------------------------------------------------
    # ---------------------------------- Override ----------------------------------
    def reset(self):
        self.resetState()
        if self.config.enable_render:
            self.render()
        self.render_cnt = 0
        return self.getObservation()

    def updateControl(self, action):
        pass

    # 包含了一些需要子类实现的抽象方法
    @abstractmethod
    def post_physics_step(self):
        pass

    @abstractmethod
    def getChangedObjectPose(self):
        raise NotImplementedError("pubObjectPose is not implemented")

    @abstractmethod
    def checkTerminated(self):
        raise NotImplementedError("checkTerminated is not implemented")    

    @abstractmethod
    def getObservation(self):
        raise NotImplementedError("getObservation is not implemented")

    @abstractmethod
    def getPrivilegedObservation(self):
        raise NotImplementedError("getPrivilegedObservation is not implemented")

    @abstractmethod
    def getReward(self):
        raise NotImplementedError("getReward is not implemented")
    
    # ---------------------------------- Override ----------------------------------
    # ------------------------------------------------------------------------------

    def step(self, action=None): # 主要的仿真步进函数
        for _ in range(self.decimation):
            self.updateControl(action)
            mujoco.mj_step(self.mj_model, self.mj_data)

        terminated = self.checkTerminated()
        if terminated:
            self.resetState()
        
        self.post_physics_step()
        if self.config.enable_render and self.render_cnt-1 < self.mj_data.time * self.render_fps:
            self.render()

        return self.getObservation(), self.getPrivilegedObservation(), self.getReward(), terminated, {}

    def view(self):
        self.mj_data.time += self.delta_t
        self.mj_data.qvel[:] = 0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        if self.render_cnt-1 < self.mj_data.time * self.render_fps:
            self.render()