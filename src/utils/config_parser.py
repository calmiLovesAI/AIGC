import argparse
import os.path

from src.diffusion.prompt import read_civitai_generate_data
from src.utils.file_ops import get_absolute_path
from omegaconf import OmegaConf


def scientific_notation(value):
    try:
        return float(value)
    except ValueError:
        # If the value cannot be parsed as a float, try parsing it as scientific notation
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid scientific notation value: {value}")


def create_civitai_conf(civitai_data_path):
    conf = read_civitai_generate_data(civitai_data_path)
    conf = OmegaConf.create(conf)
    return conf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, required=True, help='config file path', default='')
    parser.add_argument('-bs', type=int, help='batch size', default=1)
    parser.add_argument('-nsfw', action='store_false')
    args = parser.parse_args()
    return args


def load_task_cfg(*cfgs):
    task_cfgs = [OmegaConf.create(c) for c in cfgs]
    base_cfg = OmegaConf.merge(*task_cfgs)

    args = parse_args()

    if args.cfg == "civitai":
        if 'prompt_file' in base_cfg:
            base_cfg.pop('prompt_file')
        if 'negative_prompt_file' in base_cfg:
            base_cfg.pop('negative_prompt_file')
        base_cfg = OmegaConf.merge(base_cfg, create_civitai_conf('experiments/civitai.txt'))
        base_cfg.update({'from_civitai': True})
    else:
        yaml_file_path = get_absolute_path(args.cfg)
        if os.path.isfile(yaml_file_path):
            yaml_cfg = OmegaConf.load(yaml_file_path)
            base_cfg = OmegaConf.merge(base_cfg, yaml_cfg)
        base_cfg.update({'from_civitai': False})

    # update parameters
    base_cfg.update({'nsfw': args.nsfw})
    base_cfg.update({'batch_size': args.bs})

    print("The configuration is: ")
    print(OmegaConf.to_yaml(base_cfg))
    return base_cfg
