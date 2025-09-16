import random
from typing import List

import PIL.Image as pil_image
import numpy as np
import torch
from params_proto import ParamsProto
import cv2

from discoverse.randomain.utils import (
    get_value_at_index,
    import_custom_nodes,
    add_extra_model_paths,
)

def disable_comfy_args():
    """
    注释：
    该函数用于“禁用 comfy 参数解析”，通过修改 comfy.options 中的相关函数，
    避免其对命令行参数进行处理，通常用于集成或作为模块调用时防止冲突。
    """
    import comfy.options
    def enable_args_parsing(enable=False):
        global args_parsing
        args_parsing = enable
    comfy.options.enable_args_parsing = enable_args_parsing


def image_grid(img_list: List[List[pil_image.Image]]):
    """
    注释：
    将二维列表形式的 PIL.Image 图像组合成一个大图网格。
    参数：
        img_list: 二维列表，每个元素都是 PIL.Image 对象，表示每个格子的图像。
    返回：
        grid: 一个新的 PIL.Image 对象，其内容为所有输入图像拼接成的网格。
    """
    rows = len(img_list)
    cols = len(img_list[0])

    w, h = img_list[0][0].size
    grid = pil_image.new("RGB", size=(cols * w, rows * h))

    for i, row in enumerate(img_list):
        for j, img in enumerate(row):
            grid.paste(img, box=(j * w, i * h))
    return grid


class ImageGen(ParamsProto, prefix="imagen", cli=False):
    """
    注释：
    ImageGen 类负责通过三种语义掩码生成图像，生成过程结合文本描述和深度图信息，
    利用多模型（例如 ControlNet、CLIPTextEncode 等）进行条件采样和图像生成。
    
    参数说明：
      - foreground_prompt: 前景对象的文本描述
      - background_text: 背景的文本描述
      - cone_prompt: 锥形或其他物体的文本描述（此处可以随意替换为第三个对象）
      - negative_text: 用于生成负面条件，抑制不想要的效果
      
      control_parameters 配置：
        - strength: 控制条件影响强度
        - grow_mask_amount: 背景扩展掩码的大小
        - fore_grow_mask_amount: 前景扩展掩码的大小
        - background_strength: 背景条件的强度
        - fore_strength: 前景条件的强度
    """
    width = 1080
    height = 1080
    batch_size: int = 1

    num_steps = 7
    denoising_strength = 1.0

    control_strength = 0.8

    grow_mask_amount = 0
    fore_grow_mask_amount = 0

    background_strength = 0.5
    fore_strength = 1.5

    checkpoint_path = "sd_xl_turbo_1.0_fp16.safetensors"
    control_path = "controlnet_depth_sdxl_1.0.safetensors"
    vae_path = "sdxl_vae.safetensors"
    device = "cuda"

    def __post_init__(self):
        """
        注释：
        对 ImageGen 类进行初始化，加载各个必要模型与自定义节点，并完成依赖的设置。
        执行的步骤包括：
          1. 禁用 comfy 的参数解析.
          2. 添加额外的模型路径（使用 add_extra_model_paths）。
          3. 导入自定义节点（import_custom_nodes）。
          4. 加载 checkpoint 模型、文本编码模型、空白潜在图像生成节点等。
          5. 根据节点映射初始化采样器、 ControlNet 加载器、VAE 解码器和掩码处理节点。
        """
        disable_comfy_args()
        add_extra_model_paths()
        import_custom_nodes()

        from nodes import (
            EmptyLatentImage,
            CheckpointLoaderSimple,
            NODE_CLASS_MAPPINGS,
            VAEDecode,
            CLIPTextEncode,
            ControlNetLoader,
        )

        # 加载 checkpoint 模型
        checkpointloadersimple = CheckpointLoaderSimple()
        self.checkpoint = checkpointloadersimple.load_checkpoint(ckpt_name=self.checkpoint_path)
        # 加载 CLIP 文本编码节点，用于将文本描述转换为编码向量
        self.clip_text_encode = CLIPTextEncode()
        # 空白潜在图像生成节点，用于生成初始噪声图像
        self.empty_latent = EmptyLatentImage()

        # 初始化自定义采样器
        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        self.ksampler = ksamplerselect.get_sampler(sampler_name="lcm")

        # 加载 ControlNet 模型
        controlnetloader = ControlNetLoader()
        self.controlnet = controlnetloader.load_controlnet(control_net_name=self.control_path)

        # 初始化图像到掩码、掩码扩展和 VAE 解码进程的节点
        self.imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        self.growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
        self.vaedecode = VAEDecode()

        print("loading is done.")

    def generate(
            self,
            _deps=None,
            *,
            depth,
            masks,
            prompt,
            **deps,
    ):
        """
        注释：
        核心图像生成函数，根据输入的深度图、各对象掩码以及文本描述生成最终图像。
        主要工作流程：
          1. 根据输入掩码转为 torch.Tensor 格式（归一化到 [0,1]），并重复扩展以符合输入要求。
          2. 利用空白潜在图像生成节点生成初始噪声图像。
          3. 对每个对象（前景和背景）使用 CLIP 文本编码节点生成条件向量。
          4. 利用图像到掩码节点和扩展掩码节点处理输入掩码，根据配置扩展掩码区域。
          5. 将各对象条件通过 ConditioningSetMask 节点进行处理，并通过 ConditioningCombine 节点合成最终条件。
          6. 使用 ControlNetApply 节点根据深度图对合成条件进行控制。
          7. 使用采样器产生噪声采样结果，并通过 VAE 解码节点生成 RGB 图像。
          8. 对生成的图像进行后处理（转换成 BGR 格式，以便于 OpenCV 显示或保存）。

        参数：
          depth: 输入深度图（RGB 格式，后面转换为灰度图）
          masks: 包含各对象掩码的字典，键名包括 'background' 以及其它前景对象名称
          prompt: 一个字典，包含各对象的文本描述，键包括前景对象以及 'background' 和 'negative'
          _deps, **deps: 其他依赖参数（用于节点状态传播，可选）

        返回：
          生成的图像（OpenCV 格式的 BGR 图像，数据类型 uint8）。
        """
        from nodes import (
            ConditioningSetMask,
            ConditioningCombine,
            NODE_CLASS_MAPPINGS,
            ControlNetApply,
        )

        # 利用 _update 更新类的状态或依赖项（通常处理命令行或上下文依赖）
        ImageGen._update(_deps, **deps)

        # fore_objs 过滤掉背景掩码，其余所有键视为前景对象
        fore_objs = [obj for obj in list(masks.keys()) if obj != 'background']

        # 将输入的各个掩码转换为 torch.Tensor 格式，并归一化到 [0,1]
        masks_t = {}
        for k, mask in masks.items():
            # 转为灰度图
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            # 转 tensor 后扩展维度以适应后续计算要求（增加 batch 和通道维度）
            mask_t = (torch.Tensor(mask)/255.0)[None, ..., None].repeat([1, 1, 1, 3])
            masks_t[k] = mask_t
        # 同样处理深度图，转换为灰度图后扩展维度
        depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
        depth_t = (torch.Tensor(depth)/255.0)[None, ..., None].repeat([1, 1, 1, 3])

        # 利用无梯度模式确保计算过程中不记录梯度（加速推理）
        with torch.inference_mode():
            # 生成初始噪声潜在图像
            emptylatentimage = self.empty_latent.generate(
                width=ImageGen.width,
                height=ImageGen.height,
                batch_size=ImageGen.batch_size,
            )

            # 对每个对象（前景+背景+负面）生成文本编码
            textencodes = {}
            for obj in (fore_objs + ["background", "negative"]):
                textencodes[obj] = self.clip_text_encode.encode(
                    text=prompt[obj],
                    clip=get_value_at_index(self.checkpoint, 1)
                )

            # 初始化用于处理掩码与条件的相关节点
            conditioningsetmask = ConditioningSetMask()
            conditioningcombine = ConditioningCombine()
            controlnetapply = ControlNetApply()
            # 使用 SD Turbo Scheduler 生成 denoising 参数（sigmas）
            sdturboscheduler = NODE_CLASS_MAPPINGS["SDTurboScheduler"]()
            samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()

            # 根据每个前景对象和背景构建条件
            conditions = {}
            for obj in (fore_objs + ['background']):
                # 根据对象类型选择不同的掩码扩展参数
                expand = ImageGen.grow_mask_amount if obj == 'background' else ImageGen.fore_grow_mask_amount
                # 将图像转换为掩码，取通道 red（这里使用 get_value_at_index 保证数据格式）
                image2mask = self.imagetomask.image_to_mask(channel="red", image=get_value_at_index([masks_t[obj]], 0))
                # 扩展掩码范围，以增强条件控制区域
                growmask = self.growmask.expand_mask(
                    expand=expand,
                    tapered_corners=True,
                    mask=get_value_at_index(image2mask, 0)
                )
                # 根据对象类型设置对应的条件强度（背景与前景不同）
                strength = ImageGen.background_strength if obj == 'background' else ImageGen.fore_strength
                # 组合文本编码和掩码生成条件
                conditions[obj] = conditioningsetmask.append(
                    strength=strength,
                    set_cond_area="default",
                    conditioning=get_value_at_index(textencodes[obj], 0),
                    mask=get_value_at_index(growmask, 0),
                )

            # 将各条件合并成一个综合条件，逐步使用 ConditioningCombine 节点迭代合并
            conditions_list = list(conditions.values())
            final_combine = conditioningcombine.combine(
                conditioning_1=get_value_at_index(conditions_list[0], 0),
                conditioning_2=get_value_at_index(conditions_list[1], 0),
            )
            for condition in conditions_list[2:]:
                final_combine = conditioningcombine.combine(
                    conditioning_1=get_value_at_index(final_combine, 0),
                    conditioning_2=get_value_at_index(condition, 0),
                )

            # 使用 ControlNetApply 节点将合成条件与深度图结合，生成带有控制信息的条件
            controlnetapply = controlnetapply.apply_controlnet(
                strength=ImageGen.control_strength,
                conditioning=get_value_at_index(final_combine, 0),
                control_net=get_value_at_index(self.controlnet, 0),
                image=get_value_at_index((depth_t,), 0),
            )

            # 生成 denoising 参数（sigmas）
            sdturboscheduler = sdturboscheduler.get_sigmas(
                steps=ImageGen.num_steps,
                denoise=ImageGen.denoising_strength,
                model=get_value_at_index(self.checkpoint, 0),
            )

            # 利用自定义采样器采样潜在空间，结合正条件和负条件生成采样结果
            samplercustom = samplercustom.sample(
                add_noise=True,
                noise_seed=random.randint(1, 2 ** 64),
                cfg=1,
                model=get_value_at_index(self.checkpoint, 0),
                positive=get_value_at_index(controlnetapply, 0),
                negative=get_value_at_index(textencodes['negative'], 0),
                sampler=get_value_at_index(self.ksampler, 0),
                sigmas=get_value_at_index(sdturboscheduler, 0),
                latent_image=get_value_at_index(emptylatentimage, 0),
            )

            # 利用 VAE 解码器将采样结果解码还原为图像
            (image_batch,) = self.vaedecode.decode(
                samples=get_value_at_index(samplercustom, 0),
                vae=get_value_at_index(self.checkpoint, 2),
            )[:1]

            # 获取生成的图像（batch 中仅一张图）
            (generated_image,) = image_batch

            # 将生成图像乘以 255 还原到像素值范围，并转换成 uint8 类型
            gen_np = (generated_image * 255).cpu().numpy().astype("uint8")
            # 转换颜色格式从 RGB 到 BGR，以便于 OpenCV 显示或保存
            gen_np = cv2.cvtColor(gen_np, cv2.COLOR_RGB2BGR)
            return gen_np