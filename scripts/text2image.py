from experiments.config import project_cfg, txt2img_cfg
from src.diffusion.stable_diffusion_v1_5 import get_stable_diffusion_v1_5_output
from tools.config_parser import load_task_cfg
from tools.data.file_ops import get_absolute_path
from tools.platform.device import get_device


def read_prompt(file_path):
    file_path = get_absolute_path(file_path)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Removing leading and trailing empty lines and spaces
    start = 0
    end = len(lines) - 1

    while start < len(lines) and lines[start].strip() == '':
        start += 1

    while end >= 0 and lines[end].strip() == '':
        end -= 1

    prompt = ''.join(lines[start:end + 1])

    return prompt


if __name__ == '__main__':
    cfg = load_task_cfg(project_cfg, txt2img_cfg)
    device = get_device(cfg.device)
    prompt = read_prompt(cfg.prompt_file)
    print(f"The prompt is \n{prompt}")
    get_stable_diffusion_v1_5_output(prompt,
                                     model_id=cfg.model,
                                     batch_size=cfg.batch_size,
                                     scheduler_name=cfg.scheduler,
                                     num_inference_steps=cfg.num_inference_steps,
                                     device=device)
