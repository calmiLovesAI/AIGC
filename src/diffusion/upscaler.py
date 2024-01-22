import torch

from src.utils.file_ops import get_absolute_path
from src.vision.real_esrgan.real_esrgan import RealESRGAN

REAL_ESRGAN_PATH = {
    '2x': './downloads/upscaler/RealESRGAN_x2.pth',
    '4x': './downloads/upscaler/RealESRGAN_x4.pth',
    '8x': './downloads/upscaler/RealESRGAN_x8.pth',
}

ALL_UPSCALERS = [
    'Real ESRGAN'
]


def upscale_image(images, model, scale_factor=2, device=torch.device('cuda')):
    assert scale_factor in [2, 4, 8], "scale_factor must be either 2, 4, or 8"
    sr_images = []
    if model == 'Real ESRGAN':
        upscaler_model = get_absolute_path(relative_path=REAL_ESRGAN_PATH[f"{scale_factor}x"])
        model = RealESRGAN(device, scale=scale_factor)
        model.load_weights(model_path=upscaler_model)
        for img in images:
            sr_images.append(model.predict(lr_image=img))
    else:
        raise ValueError(f"{model} is not supported, only these algorithms are supported: {ALL_UPSCALERS}.")
    return sr_images
