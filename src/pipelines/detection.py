import os

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image

from src.pipelines.pipeline import AbstractPipeline
from src_old.utils.file_ops import get_absolute_path
from src_old.vision.detection_2d import get_image_paths


class ObjectDetection2dPipeline(AbstractPipeline):
    model_zoo = {
        "hustvl/yolos-tiny": "huggingface",
    }

    def __init__(self, model_name, threshold: float = 0.5, device: torch.device = torch.device("cuda")):
        """
        :param model_name:
        :param threshold: score threshold to keep object detection predictions.
        :param device:
        """
        self.threshold = threshold
        self.device = device
        if model_name not in ObjectDetection2dPipeline.model_zoo:
            raise ValueError(f"{model_name} not found in the model_zoo.")
        self.model_name = model_name
        self.model, self.image_processor = self._init_model()

    def _init_model(self):
        if ObjectDetection2dPipeline.model_zoo[self.model_name] == "huggingface":
            model, image_processor = self._load_huggingface_model()
        else:
            model, image_processor = self._load_other_model()
        return model, image_processor

    def _load_huggingface_model(self):
        model = AutoModelForObjectDetection.from_pretrained(pretrained_model_name_or_path=self.model_name)
        image_processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path=self.model_name)
        return model, image_processor

    def _load_other_model(self):
        raise NotImplementedError

    def predict(self, input_file_or_files, input_file_type: str = 'img'):
        input_file_or_files = get_absolute_path(relative_path=input_file_or_files)
        if input_file_type == "img":
            self._detect_img(input_file_or_files)
        elif input_file_type == "video":
            pass
        else:
            raise ValueError(f"File type: {input_file_type} is not supported.")

    def _detect_img(self, input_img_or_imgs):
        if os.path.isdir(input_img_or_imgs):
            # multiple pictures.
            image_paths = get_image_paths(input_img_or_imgs)
        else:
            # single picture
            image_paths = [input_img_or_imgs]
        for img_path in image_paths:
            # loop through all images
            image = Image.open(fp=img_path)
            model_inputs = self.image_processor(images=image, return_tensors='pt')
            outputs = self.model(**model_inputs)
            # convert outputs (bounding boxes and class logits) to format (xmin, ymin, xmax, ymax)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.image_processor.post_process_object_detection(outputs, threshold=self.threshold,
                                                                         target_sizes=target_sizes)[0]
            print(results)

    def _detect_video(self):
        raise NotImplementedError


def draw_detection2d_results():
    pass
