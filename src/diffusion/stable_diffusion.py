import os

from diffusers import StableDiffusionPipeline

from tools.data.file_ops import get_absolute_path


def get_diffusion_model_ckpt(file_path):
    if os.path.isfile(file_path):
        return get_absolute_path(file_path)
    return file_path


def build_stable_diffusion_model(pretrained_model, device, requires_safety_checker=True):
    pipeline = StableDiffusionPipeline.from_single_file(pretrained_model,
                                                        use_safetensors=True,
                                                        load_safety_checker=requires_safety_checker).to(device)
    return pipeline
