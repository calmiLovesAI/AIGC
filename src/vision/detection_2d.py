import glob

from transformers import pipeline
import os

from tools.data.file_ops import get_project_root, create_directory_if_not_exists
from tools.data.image import save_image
from tools.platform.device import get_device
from tools.visualize.display import display_detection_results_on_image


class Detection2D:
    def __init__(self, cfg):
        self.root_dir = get_project_root()
        self.device = get_device(type=cfg['device'])

    def _handle_image(self, model, image_path, output_path):
        detector = pipeline(model=model, device=self.device)
        if os.path.isdir(image_path):
            image_paths = get_image_paths(image_path)
            for img in image_paths:
                Detection2D._handle_single_image_and_save(detector, img, output_path)
        elif os.path.isfile(image_path):
            Detection2D._handle_single_image_and_save(detector, image_path, output_path)
        else:
            raise ValueError(f"{image_path} should be one of 'file' or 'folder'.")

    @staticmethod
    def _handle_single_image_and_save(detector, image_path, save_path):
        outputs = detector(image_path)
        image_with_detections = display_detection_results_on_image(image=image_path, detection_results=outputs)
        save_image(image=image_with_detections, save_folder=save_path)

    def _handle_video(self, model, video_path, output_path):
        pass

    def predict(self, model, file_type, file_path, output_path):
        """
        Use the model to process the file and get the prediction results.
        :param model: str, name of model
        :param file_type: str, can be 'img', 'video'
        :param file_path: str, a specific file path or folder
        :param output_path: str, the path where the results will be saved.
        :return:
        """
        # get absolute path
        file_path = os.path.join(self.root_dir, file_path)
        output_path = os.path.join(self.root_dir, output_path)
        create_directory_if_not_exists(output_path)
        match file_type:
            case 'img':
                self._handle_image(model, file_path, output_path)
            case 'video':
                self._handle_video(model, file_path, output_path)
            case _:
                raise ValueError(f"{file_type} is not supported.")

    def train(self):
        pass

    def evaluate(self):
        pass


def get_image_paths(image_path):
    """
    Get all image file paths under the image_path path.
    :param image_path:
    :return:
    """
    image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
    image_pattern = os.path.join(image_path, f'*.*')  # match all files under the specified path

    # Use the glob module to get all qualified file paths under a specified path.
    image_files = glob.glob(image_pattern)

    # Filter out file paths that match image file extensions.
    image_paths = [file for file in image_files if file.lower().split('.')[-1] in image_extensions]

    return image_paths


def get_video_paths(video_path):
    """
    Get all video file paths under the video_path path.
    :param video_path:
    :return:
    """
    # Supported video file extensions
    video_extensions = ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'mpeg',
                        'mpg']  # Add more extensions if needed

    # Constructing the pattern to match video files
    video_pattern = os.path.join(video_path, '*.*')

    # Getting all files in the specified path
    video_files = glob.glob(video_pattern)

    # Filtering the files based on video extensions
    video_paths = [file for file in video_files if file.lower().split('.')[-1] in video_extensions]

    return video_paths
