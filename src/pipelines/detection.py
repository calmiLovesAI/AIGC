import os
from typing import Dict

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection, PreTrainedModel
from PIL import Image, ImageDraw

from src.colors import predefined_colors
from src.models.yolo import load_ultralytics_model, yolo_detect
from src.pipelines.pipeline import AbstractPipeline
from src.save import save_image
from src_old.utils.file_ops import get_absolute_path
from src_old.vision.detection_2d import get_image_paths


class ObjectDetection2dPipeline(AbstractPipeline):
    huggingface_model: Dict[str, PreTrainedModel] = {
        "hustvl/yolos-tiny": AutoModelForObjectDetection,
        "hustvl/yolos-small": AutoModelForObjectDetection,
        "hustvl/yolos-base": AutoModelForObjectDetection,
        "facebook/detr-resnet-50": AutoModelForObjectDetection,
        "facebook/detr-resnet-101": AutoModelForObjectDetection,
        "SenseTime/deformable-detr": AutoModelForObjectDetection,
        "microsoft/conditional-detr-resnet-50": AutoModelForObjectDetection,
    }

    other_model = {
        "yolo_8_n": "ultralytics",
        "yolo_8_s": "ultralytics",
        "yolo_8_m": "ultralytics",
        "yolo_8_l": "ultralytics",
        "yolo_8_x": "ultralytics",
    }

    def __init__(self, model_name: str, threshold: float = 0.5,
                 device: torch.device = torch.device("cuda")):
        """
        :param model_name:
        :param threshold: score threshold to keep object detection predictions.
        :param device:
        """
        model_name = model_name.lower()
        model_zoo = {**ObjectDetection2dPipeline.huggingface_model, **ObjectDetection2dPipeline.other_model}
        self.threshold = threshold
        self.device = device
        if model_name not in model_zoo:
            raise ValueError(f"{model_name} not found in the model_zoo.")
        self.model_name = model_name
        if model_name in ObjectDetection2dPipeline.huggingface_model:
            self.model_source = "huggingface"
        else:
            self.model_source = ObjectDetection2dPipeline.other_model[model_name]
        self.model, self.image_processor = self._init_model()

    def _init_model(self):
        if self.model_source == "huggingface":
            model, image_processor = self._load_huggingface_model()
        else:
            model, image_processor = self._load_other_model()
        return model, image_processor

    def _load_huggingface_model(self):
        # The model is set in evaluation mode by default using model.eval()(so for instance, dropout modules are deactivated).
        # To train the model, you should first set it back in training mode with model.train()
        model = AutoModelForObjectDetection.from_pretrained(pretrained_model_name_or_path=self.model_name)
        image_processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path=self.model_name)
        return model, image_processor

    def _load_other_model(self):
        if self.model_source == "ultralytics":
            model = load_ultralytics_model(model_name=self.model_name)
        return model, None

    def predict(self, input_file_or_files, input_file_type: str = 'img', save_result: bool = True):
        # Convert relative paths to absolute paths.
        input_file_or_files = get_absolute_path(relative_path=input_file_or_files)
        if input_file_type == "img":
            self._detect_img(input_file_or_files, save_result)
        elif input_file_type == "video":
            pass
        else:
            raise ValueError(f"File type: {input_file_type} is not supported.")

    def _detect_img(self, input_img_or_imgs, save_result):
        if os.path.isdir(input_img_or_imgs):
            # multiple pictures.
            image_paths = get_image_paths(input_img_or_imgs)
        else:
            # single picture
            image_paths = [input_img_or_imgs]
        if self.model_source == "huggingface":
            for img_path in image_paths:
                # loop through all images
                detection_results = hugging_face_model_detection_2d(img_path, self.image_processor, self.model,
                                                                    self.threshold)
                print(detection_results)
                if save_result:
                    detected = draw_detection2d_results(img_path, detection_results,
                                                        id2label=self.model.config.id2label)
                    save_image(image=detected, filename=self.model_name)

        if self.model_source == "ultralytics":
            detection_results, id2label = yolo_detect(model=self.model, input_data=image_paths, conf=self.threshold)
            for i in range(len(image_paths)):
                detected = draw_detection2d_results(image_paths[i], detection_results[i], id2label=id2label)
                save_image(image=detected, filename=self.model_name)

    def _detect_video(self):
        raise NotImplementedError


def hugging_face_model_detection_2d(img_path, image_processor, model, threshold):
    image = Image.open(fp=img_path)
    model_inputs = image_processor(images=image, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**model_inputs)
    # convert outputs (bounding boxes and class logits) to format (xmin, ymin, xmax, ymax)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=threshold,
                                                            target_sizes=target_sizes)[0]
    detection_results = {
        'boxes': results['boxes'].cpu().numpy(),  # (N, 4)  float32
        'labels': results['labels'].cpu().numpy(),  # (N,)  int64
        'scores': results['scores'].cpu().numpy()  # (N,)  float32
    }
    return detection_results


def draw_detection2d_results(image, detection_results, id2label):
    """
    Display the results of object detection on the picture.
    :param image: image path or image read through PIL.Image
    :param detection_results: The result of object vision has a similar format:
    [{'scores': confidence, np.ndarray, shape is (N,),
      'labels': category id, np.ndarray, shape is (N,),
      'boxes':  [[xmin, ymin, xmax, ymax], [...], ...], np.ndarray, shape is (N, 4)
    :param id2label: dict
    :return: image with detection results
    """
    if isinstance(image, str):
        image = Image.open(image)

    draw = ImageDraw.Draw(image)

    scores, labels, boxes = detection_results['scores'], detection_results['labels'], detection_results['boxes'].astype(
        np.int32)
    n_boxes = len(boxes)
    for i in range(n_boxes):
        label_id = labels[i]
        category = id2label[label_id]
        score = scores[i]
        xmin, ymin, xmax, ymax = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

        color = predefined_colors[label_id % len(predefined_colors)]
        draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=2)
        draw.text((xmin, ymin), f"{category}: {score:.2f}", fill='blue')

    return image
