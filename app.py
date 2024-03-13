import argparse

from src.pipelines.detection import ObjectDetection2dPipeline
from src.pipelines.segmentation import Segmentation2dPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True, help='task name')
    parser.add_argument('-m', '--model', type=str, required=True, help='model name')
    parser.add_argument('-f', '--file', type=str, required=True, help='input file or root of files')

    parser.add_argument('-th', '--threshold', type=float,
                        help='object detection threshold (to keep object detection predictions)',
                        default=0.7)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.task.lower() == "det2d":
        pipe = ObjectDetection2dPipeline(model_name=args.model, threshold=args.threshold)
    if args.task.lower() == "seg2d":
        pipe = Segmentation2dPipeline(model_name=args.model)
    pipe.predict(input_file_or_files=args.file)
