from experiments.config import project_cfg, detection_cfg
from src.vision.detection_2d import Detection2D
from src.utils.config_parser import load_task_cfg
from src.utils.device import get_device

if __name__ == '__main__':
    cfg = load_task_cfg(project_cfg, detection_cfg)
    device = get_device(cfg.device)
    detection2d_pipeline = Detection2D(device=device)
    detection2d_pipeline.predict(model=cfg.model, file_type='img', file_path='./test_samples/images/',
                                 output_path='./test_samples/results/')

