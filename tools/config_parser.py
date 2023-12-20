import argparse
import json
import os.path

import yaml
from collections import namedtuple

from tools.data.file_ops import get_file_extension


def load_cfg_from_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            cfg_data = yaml.safe_load(file)
            return cfg_data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return {}


def load_cfg_from_json(file_path):
    try:
        with open(file_path, 'r') as file:
            cfg_data = json.load(file)
            return cfg_data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return {}


def merge_dicts(*args):
    merged_dict = {}
    for dictionary in args:
        merged_dict.update(dictionary)
    return merged_dict


def merge_from_file(a, file_path):
    """
    Merge the current configuration dictionary a with the configuration dictionary read from the file.
    :param a:
    :param file_path:
    :return:
    """
    extension = get_file_extension(file_path)
    if extension.lower() == '.json':
        cfg = load_cfg_from_json(file_path)
        return a.update(cfg)
    elif extension.lower() in ('.yaml', '.yml'):
        cfg = load_cfg_from_yaml(file_path)
        return a.update(cfg)
    else:
        return ValueError(f"Unsupported file type: {extension}.")


def dict_to_namedtuple(a):
    # Extract keys from the dictionary to use as field names for the named tuple
    fields = list(a.keys())

    # Create a named tuple class
    MyTuple = namedtuple('MyTuple', fields)

    # Convert the dictionary to a named tuple and return
    return MyTuple(**a)


def load_task_cfg(*cfgs):
    task_cfg = merge_dicts(*cfgs)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, help='config file path', default='')
    args = parser.parse_args()

    if os.path.exists(args.cfg):
       task_cfg = merge_from_file(task_cfg, args.cfg)
    print("The configuration is: \n", task_cfg)
    return dict_to_namedtuple(task_cfg)
