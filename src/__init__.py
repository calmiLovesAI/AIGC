import os

from experiments.config import project_cfg
from tools.data.file_ops import get_project_root

__all__ = [
    'initialize',
]


def initialize():
    cfg = load_cfg()
    # init_hugging_face(hf_home_path=os.path.join(get_project_root(), cfg["hf_home_path"]))
    return cfg


def init_hugging_face(hf_home_path):
    os.environ['HF_HOME'] = hf_home_path

    print(f"'HF_HOME' has been set to {hf_home_path}.")


def load_cfg():
    return project_cfg
