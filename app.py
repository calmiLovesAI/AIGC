import argparse

from src.pipelines.detection import ObjectDetection2dPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True, help='task name')
    parser.add_argument('-m', '--model', type=str, required=True, help='model name')
    parser.add_argument('-f', type=str, required=True, help='input file or root of files')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.task.lower() == "det2d":
        pipe = ObjectDetection2dPipeline(model_name=args.model)
    pipe.predict(input_file_or_files=args.f)
