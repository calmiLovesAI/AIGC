import cv2
import os
import numpy as np

from calminet.data.file_ops import get_project_root, generate_random_filename


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


def save_image(image, save_path=None):
    """
    Save image to file.
    :param image: numpy.ndarray，image that read by opencv.
    :param save_path: None for random generated path.
    :return:
    """
    if save_path is None:
        filename = generate_random_filename("img", "jpg")
        save_path = os.path.join(get_project_root(), "test", filename)
    cv2.imwrite(save_path, image)
    print(f"Saved to {save_path}.")
