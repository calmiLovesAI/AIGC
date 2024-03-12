import os
from datetime import date
from PIL import Image

import numpy as np
import cv2

from src_old.utils.file_ops import create_directory_if_not_exists, generate_random_filename


def save_image(image, save_folder=None, filename='img', suffix='.png'):
    """
    Save image as file.
    :param image: numpy.ndarray or PIL.Image.Image.
    :param save_folder:
    :param filename:
    :param suffix:
    :return:
    """
    filename = generate_random_filename(filename, suffix)
    if save_folder is None:
        # Get the current date
        today_date = date.today()
        # Create a folder name based on the current date
        cur_date_folder_name = today_date.isoformat()
        # Create the folder path for the current date
        save_folder = os.path.join('outputs', cur_date_folder_name)
    _ = create_directory_if_not_exists(save_folder)
    save_path = os.path.join(_, filename)
    if isinstance(image, np.ndarray):
        cv2.imwrite(save_path, image)
    elif isinstance(image, Image.Image):
        image.save(fp=save_path)
    else:
        raise ValueError(f"Unsupported image type.")
    print(f"Saved to {save_path}.")
