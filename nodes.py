import os
import torch
import random
import diffusers
import torch.utils
import unet.utils as utils
from unet.unet_controller import UNetController
import numpy as np
import folder_paths
import torch
import random
from datetime import datetime

diffusers.utils.logging.set_verbosity_error()


class PromptStoryModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                ["playground-v2.5-1024px-aesthetic"], {"default": "playgroundai/playground-v2.5-1024px-aesthetic"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["fp16", "fp32"], {"default": "fp16"}),
                "load_local_model": ("BOOLEAN", {"default": False}),
            }, "optional": {
                "local_model_path": ("STRING", {"default": "playgroundai/playground-v2.5-1024px-aesthetic"}),
            }
        }

    RETURN_TYPES = ("STORY_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "StoryGeneration"

    def load_model(self, model_name, load_local_model, device, precision, *args, **kwargs):
        if load_local_model:
            model_path = kwargs.get("local_model_path", "playgroundai/playground-v2.5-1024px-aesthetic")
        else:
            model_path = "playgroundai/playground-v2.5-1024px-aesthetic"
        pipe, _ = utils.load_pipe_from_path(
            model_path,
            device,
            torch.float16 if precision == "fp16" else torch.float32,
            precision
        )
        unet_controller = self.create_unet_controller(pipe, device)
        return ({"pipe": pipe, "controller": unet_controller},)

    def create_unet_controller(self, pipe, device):
        controller = UNetController()
        controller.device = device
        controller.tokenizer = pipe.tokenizer
        return controller


class GenerateStoryImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "story_model": ("STORY_MODEL",),
                "id_prompt": ("STRING", {"multiline": True, "default": "A photo of a red fox with coat"}),
                "frame_prompts": ("STRING", {"multiline": True,
                                             "default": "wearing a scarf in a meadow\nplaying in the snow\nat the edge of a village with river"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "window_length": ("INT", {"default": 10, "min": 1, "max": 100}),
                "save_dir": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_story"
    CATEGORY = "StoryGeneration"

    def generate_story(self, story_model, id_prompt, frame_prompts, seed, window_length,save_dir):
        # 参数解析
        frame_prompt_list = [p.strip() for p in frame_prompts.split('\n') if p.strip()]
        if not frame_prompt_list:
            raise ValueError("至少需要提供一个帧提示词")

        # 生成随机种子
        if seed == 0:
            seed = random.randint(0, 1000000)

        if not save_dir:
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            save_dir = os.path.join(folder_paths.get_output_directory(), f"story_{current_time}")
        os.makedirs(save_dir, exist_ok=True)

        # 执行生成逻辑
        images, _ = utils.movement_gen_story_slide_windows(
            id_prompt,
            frame_prompt_list,
            story_model["pipe"],
            window_length,
            seed,
            story_model["controller"],
            save_dir
        )

        # 转换图像格式为ComfyUI标准格式
        tensor_images = torch.cat([self.image_to_tensor(img) for img in images], dim=0)
        return (tensor_images,)

    def image_to_tensor(self, img):
        img = img.convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]
        return img


# 节点注册
NODE_CLASS_MAPPINGS = {
    "PromptStoryModelLoader": PromptStoryModelLoader,
    "GenerateStoryImage": GenerateStoryImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptStoryModelLoader": "PromptStoryModelLoader",
    "GenerateStoryImage": "GenerateStoryImage"
}
