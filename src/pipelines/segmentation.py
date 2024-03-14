import os
import random
from typing import Dict

import torch
from PIL import Image, ImageDraw

from src.colors import predefined_colors
from src.pipelines.pipeline import AbstractPipeline
from src.save import save_image
from src_old.utils.file_ops import get_absolute_path
from src_old.vision.detection_2d import get_image_paths
from transformers import ImageSegmentationPipeline, AutoImageProcessor, DetrForSegmentation, PreTrainedModel, \
    SegformerForSemanticSegmentation


class Segmentation2dPipeline(AbstractPipeline):
    huggingface_model: Dict[str, PreTrainedModel] = {
        "nvidia/segformer-b0-finetuned-ade-512-512": SegformerForSemanticSegmentation,
        "facebook/detr-resnet-50-panoptic": DetrForSegmentation,
    }

    other_model = {}

    def __init__(self, model_name: str,
                 threshold: float = 0.9,
                 mask_threshold: float = 0.5,
                 overlap_mask_area_threshold: float = 0.5,
                 device: torch.device = torch.device("cuda")):
        """
        Initialization method.
        :param model_name:
        :param threshold: Probability threshold to filter out predicted masks.
        :param mask_threshold: Threshold to use when turning the predicted masks into binary values.
        :param overlap_mask_area_threshold: Mask overlap threshold to eliminate small, disconnected segments.
        :param device: where the deep learning model runs.
        """
        model_name = model_name.lower()
        model_zoo = {**Segmentation2dPipeline.huggingface_model, **Segmentation2dPipeline.other_model}
        self.device = device
        if model_name not in model_zoo:
            raise ValueError(f"{model_name} not found in the model_zoo.")
        if model_name in Segmentation2dPipeline.huggingface_model:
            self.model_source = "huggingface"
        else:
            self.model_source = "other"

        self.model_name = model_name
        self.model, self.image_processor = self._init_model()

        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.overlap_mask_area_threshold = overlap_mask_area_threshold

    def _init_model(self):
        if self.model_source == "huggingface":
            # Actually, model is a huggingface pipeline.
            model, image_processor = self._load_huggingface_model()
        else:
            model, image_processor = self._load_other_model()
        return model, image_processor

    def _load_huggingface_model(self):
        # The model is set in evaluation mode by default using model.eval()(so for instance, dropout modules are deactivated).
        # To train the model, you should first set it back in training mode with model.train()
        auto_pretrained_model = Segmentation2dPipeline.huggingface_model[self.model_name]
        model = auto_pretrained_model.from_pretrained(pretrained_model_name_or_path=self.model_name)

        image_processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path=self.model_name)
        return model, image_processor

    def _load_other_model(self):
        raise NotImplementedError

    def predict(self, input_file_or_files, input_file_type: str = 'img', save_result: bool = True):
        input_file_or_files = get_absolute_path(relative_path=input_file_or_files)
        if input_file_type == "img":
            self._segment_img(input_file_or_files, save_result)
        elif input_file_type == "video":
            pass
        else:
            raise ValueError(f"File type: {input_file_type} is not supported.")

    def _segment_img(self, input_img_or_imgs, save_result):
        if os.path.isdir(input_img_or_imgs):
            # multiple pictures.
            image_paths = get_image_paths(input_img_or_imgs)
        else:
            # single picture
            image_paths = [input_img_or_imgs]
        for img_path in image_paths:
            # loop through all images
            if self.model_source == "huggingface":
                pipe = ImageSegmentationPipeline(model=self.model, image_processor=self.image_processor)
                segmentation_results = pipe(images=img_path, threshold=self.threshold,
                                            mask_threshold=self.mask_threshold,
                                            overlap_mask_area_threshold=self.overlap_mask_area_threshold)
                print(segmentation_results)

            if save_result:
                segmented = draw_segmentation2d_results(img_path, segmentation_results, 0.5)
                save_image(image=segmented, filename=self.model_name)

    def _segment_video(self):
        raise NotImplementedError


def draw_segmentation2d_results(image, segmentation_results, opacity):
    """
    Display the results of semantic segmentation on the image.
    :param image: image path or image read through PIL.Image
    :param segmentation_results: List of dict, [{'score': float, 'label': str, 'mask': PIL.Image.Image mode=L}, ...]
    :param opacity: float, the value range is 0.0~1.0, 0.0 means completely transparent, 1.0 means completely opaque.
    :return: PIL.Image.Image, image with segmentation results
    """
    if isinstance(image, str):
        image = Image.open(image)

    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    colors = random.sample(predefined_colors, len(predefined_colors))

    for i in range(len(segmentation_results)):
        score = segmentation_results[i]['score']  # float
        label = segmentation_results[i]['label']  # str
        mask = segmentation_results[i]['mask']  # PIL.Image.Image
        # Draw the mask on the image
        draw.bitmap((0, 0), mask, fill=colors[i])

    return Image.blend(image, image_copy, opacity)
