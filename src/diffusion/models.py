import os

from tools.data.file_ops import get_absolute_path


def get_diffusion_model_ckpt(file_path):
    if os.path.isfile(file_path):
        return get_absolute_path(file_path)
    return file_path