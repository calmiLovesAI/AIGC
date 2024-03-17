import os.path
import numpy as np

from ultralytics import YOLO
from src.models.urls import ULTRALYTICS
from src_old.utils.file_ops import download_file, create_directory_if_not_exists, get_absolute_path


def load_ultralytics_model(model_name: str, download_root: str = './yolo'):
    """
    load ultralytics models, see docs: https://docs.ultralytics.com/
    :param model_name: str, the name of model
    :param download_root: str, where the model should be downloaded.
    :return: ultralytics YOLO
    """
    # download model
    model_filename = model_name + ".pt"
    model_path = get_absolute_path(os.path.join('./downloads', download_root, model_filename))
    create_directory_if_not_exists(model_path, is_file=True)
    download_file(url=ULTRALYTICS[model_name], model_dir=model_path)
    # load model
    model = YOLO(model=model_path)
    return model


def yolo_detect(model, input_data, conf: float = 0.25, iou: float = 0.7, max_det: int = 300):
    """
    The post-process function of ultralytics detection models
    :param model: YOLO model
    :param input_data: image, PIL.Image, URL, video, etc.
    :param conf: float, Sets the minimum confidence threshold for detections. Objects detected with confidence below
                        this threshold will be disregarded. Adjusting this value can help reduce false positives.
    :param iou: float,  Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS).
                        Higher values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.
    :param max_det: int, Maximum number of detections allowed per image. Limits the total number of objects
                         the model can detect in a single inference, preventing excessive outputs in dense scenes.
    :return: detections: [{'boxes': xyxy format, 'labels': int64, 'scores': float32}, ...], id2label: dict
    """
    detections = []

    results = model(input_data, conf=conf, iou=iou, max_det=max_det)  # List
    id2label = results[0].names

    for result in results:
        labels = result.boxes.cls.cpu().numpy().astype(np.int64)
        scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        detections.append({
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        })
    return detections, id2label


def yolo_segment():
    pass


def yolo_pose():
    pass


def yolo_obb():
    """
    Oriented object detection goes a step further than object detection and introduce an extra angle to
    locate objects more accurate in an image.
    :return:
    """
    pass
