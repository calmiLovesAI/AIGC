import os
import time
from datetime import datetime

import torch


def get_project_root():
    current_dir = os.path.abspath(__file__)
    while not os.path.exists(os.path.join(current_dir, 'LICENSE')):
        if current_dir == os.path.dirname(current_dir):
            raise FileNotFoundError("Project root directory not found.")
        current_dir = os.path.dirname(current_dir)  # Get the parent directory of the current directory
    return current_dir


def generate_random_filename(file_prefix, suffix):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%H-%M-%S-%f")[:-3]  # Format time to milliseconds
    filename = f"{file_prefix}_{formatted_time}{suffix}"

    return filename


def create_directory_if_not_exists(directory_path, is_file=False):
    file_name = ""
    if is_file:
        directory_path, file_name = os.path.split(directory_path)
    if not os.path.isabs(directory_path):
        directory_path = get_absolute_path(directory_path)
    # Check if the directory path exists
    if not os.path.exists(directory_path):
        # Create the directory and its parent directories if they don't exist
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    return os.path.join(directory_path, file_name)


def get_absolute_path(relative_path):
    if os.path.isabs(relative_path):
        return relative_path
    root_path = get_project_root()  # Get the absolute path of the project root directory
    absolute_path = os.path.normpath(os.path.join(root_path, relative_path))  # Join paths
    return absolute_path


def get_file_extension(file_path):
    return os.path.splitext(file_path)[1]


def download_file(url, model_dir):
    """
    Download file from 'url' to 'model_dir'.
    :param url:
    :param model_dir:
    :return:
    """
    t0 = time.time()
    if os.path.exists(model_dir):
        print(f"File '{model_dir}' already exists.")
    else:
        print(f"Start downloading from: {url}")
        torch.hub.download_url_to_file(url=url, dst=model_dir)
        # r = requests.get(url, stream=True)
        # total = int(r.headers.get("content-length", 0))
        # with open(model_dir, mode="wb") as f, \
        #         tqdm(desc=model_path, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        #     for data in r.iter_content(chunk_size=1024):
        #         size = f.write(data)
        #         bar.update(size)
        print(f"Download completed, which took: {time.time() - t0:.2f}s")


def load_state_dict_from_url(url: str, model_dir: str, map_location: torch.device = None) -> dict:
    """
    Download the state dict of a pytorch model to a specific local dir.
    :param url:
    :param model_dir:
    :param map_location:
    :return:
    """
    model_dir = create_directory_if_not_exists(model_dir, is_file=True)
    download_file(url, model_dir)
    state_dict = torch.load(model_dir, map_location=map_location)
    return state_dict


def create_checkpoint_save_dir(model_name: str, dataset_name: str, root: str = 'outputs'):
    model_name = model_name.split('/')[-1].lower()
    dataset_name = dataset_name.lower()
    save_dir = os.path.join(root, f"{model_name}_{dataset_name}")
    save_dir = create_directory_if_not_exists(directory_path=save_dir)
    return save_dir
