import argparse

from src_old.diffusion.stable_video_diffusion import generate_video_with_svd_xt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', type=str, required=True, help='image path or url')
    parser.add_argument('-seed', type=int, help='random seed', default=-1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_video_with_svd_xt(
        condition_image_path_or_url=args.img,
        seed=args.seed
    )
