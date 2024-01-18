import argparse
import os.path

from src.utils.file_ops import get_absolute_path
from omegaconf import OmegaConf


def load_task_cfg(*cfgs):
    task_cfgs = [OmegaConf.create(c) for c in cfgs]
    base_cfg = OmegaConf.merge(*task_cfgs)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, help='config file path', default='')
    args = parser.parse_args()
    yaml_file_path = get_absolute_path(args.cfg)

    if os.path.isfile(yaml_file_path):
        yaml_cfg = OmegaConf.load(yaml_file_path)
        base_cfg = OmegaConf.merge(base_cfg, yaml_cfg)
    print("The configuration is: ")
    print(OmegaConf.to_yaml(base_cfg))
    return base_cfg
