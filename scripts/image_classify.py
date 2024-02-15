import argparse

from src.computer_vision.image_classification import train_image_classification_model, do_image_classification
from src.utils.config_parser import scientific_notation
from src.utils.file_ops import create_checkpoint_save_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, help='mode: train or infer', default='train')
    parser.add_argument('-model', type=str, required=True, help='model name')
    parser.add_argument('-data', type=str, required=True, help='dataset')
    parser.add_argument('-ckpt', type=str, help='checkpoint path', default='')
    parser.add_argument('-epoch', type=int)
    parser.add_argument('-lr', type=scientific_notation, help='learning rate', default=5e-5)
    parser.add_argument('-bs', type=int, help='batch size', default=8)
    parser.add_argument('-metric', type=str, help='metric name')
    parser.add_argument('-test_img', type=str, help='test image path', default='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.m == 'train':
        train_image_classification_model(
            model_id=args.model,
            dataset=args.data,
            output_dir=create_checkpoint_save_dir(
                model_name=args.model,
                dataset_name=args.data
            ),
            epochs=args.epoch,
            train_batch_size=args.bs,
            eval_batch_size=args.bs,
            learning_rate=args.lr,
        )
    elif args.m == 'infer':
        do_image_classification(
            checkpoint_path=args.ckpt,
            image_path=args.test_img,
        )
    else:
        raise ValueError
