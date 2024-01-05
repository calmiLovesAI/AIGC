import os
import torch

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from tools.data.file_ops import get_absolute_path


def build_stable_diffusion_pipeline(pretrained_model, requires_safety_checker=True, device='cuda'):
    """
    Build a stable diffusion 1.5 pipeline
    :param pretrained_model: str or os.PathLike, A link to the .ckpt file on the hub or a path to a file containing all pipeline weights.
    :param requires_safety_checker: bool
    :param device:
    :return:
    """
    if os.path.isfile(pretrained_model):
        pretrained_model = get_absolute_path(pretrained_model)
    pipeline = StableDiffusionPipeline.from_single_file(pretrained_model,
                                                        use_safetensors=True,
                                                        load_safety_checker=requires_safety_checker).to(device)
    try:
        pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
    except Exception as e:
        print(e)
    return pipeline


def stable_diffusion_forward():
    pass


def build_stable_diffusion_xl_pipeline():
    pass


def stable_diffusion_xl_forward():
    pass
