import glob

from transformers import pipeline
import os

from src.utils.file_ops import create_directory_if_not_exists, get_absolute_path
from src.utils.file_type import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from src.utils.image import save_image
from src.utils.vis import display_detection_results_on_image


class Detection2D:
    def __init__(self, device):
        self.device = device

    def _build_detector(self, model):
        return pipeline(model=model, device=self.device)

    def _handle_image(self, detector, image_path, output_path):
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

    def _handle_video(self, detector, video_path, output_path):
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
        file_path = get_absolute_path(file_path)
        output_path = create_directory_if_not_exists(output_path)

        detector = self._build_detector(model)

        match file_type:
            case 'img':
                self._handle_image(detector, file_path, output_path)
            case 'video':
                self._handle_video(detector, file_path, output_path)
            case _:
                raise ValueError(f"{file_type} is not supported.")

    def train(self, model, dataset):
        """
        :param model: str
        :param dataset: str
        :return:
        """
        pass

    def evaluate(self):
        pass


def get_files(root_path, extensions):
    file_pattern = os.path.join(root_path, f'*.*')  # match all files under the specified path

    # Use the glob module to get all qualified file paths under a specified path.
    files = glob.glob(file_pattern)

    # Filter out file paths that match image file extensions.
    out_file_paths = [file for file in files if file.lower().split('.')[-1] in extensions]

    return out_file_paths


def get_image_paths(image_path):
    """
    Get all image file paths under the image_path path.
    :param image_path:
    :return:
    """
    return get_files(root_path=image_path, extensions=IMAGE_EXTENSIONS)


def get_video_paths(video_path):
    """
    Get all video file paths under the video_path path.
    :param video_path:
    :return:
    """
    return get_files(root_path=video_path, extensions=VIDEO_EXTENSIONS)
