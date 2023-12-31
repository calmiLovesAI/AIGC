import os
from datetime import datetime


def get_project_root():
    current_dir = os.path.abspath(__file__)
    while not os.path.exists(os.path.join(current_dir, 'LICENSE')):
        if current_dir == os.path.dirname(current_dir):
            raise FileNotFoundError("Project root directory not found.")
        current_dir = os.path.dirname(current_dir)  # Get the parent directory of the current directory
    return current_dir


def generate_random_filename(file_type, suffix):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%H-%M-%S-%f")[:-3]  # Format time to milliseconds
    match file_type:
        case "img":
            filename = f"{file_type}_{formatted_time}{suffix}"
        case _:
            filename = f"{file_type}_{formatted_time}{suffix}"
    return filename


def create_directory_if_not_exists(directory_path):
    if not os.path.isabs(directory_path):
        directory_path = get_absolute_path(directory_path)
    # Check if the directory path exists
    if not os.path.exists(directory_path):
        # Create the directory and its parent directories if they don't exist
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    return directory_path


def get_absolute_path(relative_path):
    if os.path.isabs(relative_path):
        return relative_path
    root_path = get_project_root()  # Get the absolute path of the project root directory
    absolute_path = os.path.normpath(os.path.join(root_path, relative_path))  # Join paths
    return absolute_path


def get_file_extension(file_path):
    return os.path.splitext(file_path)[1]


