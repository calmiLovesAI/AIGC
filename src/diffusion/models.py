import os

from diffusers import StableDiffusionPipeline

from tools.data.file_ops import get_absolute_path


def get_diffusion_model_ckpt(file_path):
    if os.path.isfile(file_path):
        return get_absolute_path(file_path)
    return file_path


def build_stable_diffusion_model(pretrained_model, device, requires_safety_checker=True):
    pipeline = StableDiffusionPipeline.from_single_file(pretrained_model_link_or_path=pretrained_model,
                                                        use_safetensors=True).to(device)
    if not requires_safety_checker:
        pipeline.safety_checker = None
        pipeline.requires_safety_checker = False
    return pipeline
