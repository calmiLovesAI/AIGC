import os

from diffusers import StableDiffusionPipeline

from tools.data.file_ops import get_absolute_path


def get_diffusion_model_ckpt(file_path):
    if os.path.isfile(file_path):
        return get_absolute_path(file_path)
    return file_path


def build_stable_diffusion_model(pretrained_model, device, requires_safety_checker=True):
    safety_param = {}
    if not requires_safety_checker:
        safety_param.update({
            'safety_checker': None
        })
    pipeline = StableDiffusionPipeline.from_single_file(pretrained_model,
                                                        use_safetensors=True,
                                                        **safety_param).to(device)
    # if not requires_safety_checker:
    #     pipeline.safety_checker = None
    #     pipeline.requires_safety_checker = False
    return pipeline
