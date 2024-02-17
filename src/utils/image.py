from datetime import date

import cv2
import torch
import os
import numpy as np
from PIL import Image

from src.utils.file_ops import generate_random_filename, create_directory_if_not_exists
from src.diffusion.prompt import get_filename_from_prompt


def read_image(image_path, mode='rgb'):
    """
    Read image from file.
    :param image_path: iamge file path
    :param mode: str, image format，'rgb', 'bgr' and 'gray' are supported
    :return: numpy.ndarray, dtype=np.uint8, shape=(h, w, c)
    """
    assert mode in ['rgb', 'bgr', 'gray'], "mode must be one of 'rgb', 'bgr', 'gray'"
    image_array = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if mode == "rgb":
        return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    elif mode == 'gray':
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, axis=-1)
        return gray
    else:
        # default "bgr"
        return image_array


def save_image(image, save_folder, filename=None):
    """
    Save image to file.
    :param image: numpy.ndarray，image that read by opencv.
    :param save_folder:
    :param filename:
    :return:
    """
    if not filename:
        filename = generate_random_filename("img", ".png")
    _ = create_directory_if_not_exists(save_folder)
    save_path = os.path.join(_, filename)
    if isinstance(image, np.ndarray):
        cv2.imwrite(save_path, image)
    elif isinstance(image, Image.Image):
        image.save(fp=save_path)
    else:
        raise ValueError(f"Unsupported image type.")
    print(f"Saved to {save_path}.")


def decode_denoised_output(input):
    """
    convert the denoised output into an image
    :param input: torch.Tensor, sahpe=torch.Size([1, 3, h, w]), dtype=torch.float32
    :return:
    """
    image = (input / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    return image


def save_ai_generated_image(image, seed, save_folder, prompt='none'):
    """
    Save the image generated by AI, and the image name is related to prompt.
    :param image: PIL.Image.Image,
    :param seed: int, random seed.
    :param save_folder: str, a relative path or an absolute path.
    :param prompt: str,
    :return:
    """
    assert isinstance(image, Image.Image)

    # Get the current date
    today_date = date.today()
    # Create a folder name based on the current date
    cur_date_folder_name = today_date.isoformat()
    # Create the folder path for the current date
    cur_date_folder_path = os.path.join(save_folder, cur_date_folder_name)

    save_dir = create_directory_if_not_exists(cur_date_folder_path)

    filename_prefix = get_filename_from_prompt(prompt, length=50)
    filename = generate_random_filename(file_prefix=filename_prefix, suffix=f"_seed={seed}.png")

    save_dir = os.path.join(save_dir, filename)
    image.save(fp=save_dir)
    print(f"Saved to {save_dir}.")
