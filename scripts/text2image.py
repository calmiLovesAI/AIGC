from experiments.config import project_cfg, txt2img_cfg
from src.diffusion.prompt import read_prompt
from src.diffusion.generator import Text2ImageGenerator
from tools.config_parser import load_task_cfg
from tools.platform.device import get_device

if __name__ == '__main__':
    cfg = load_task_cfg(project_cfg, txt2img_cfg)
    device = get_device(cfg.device)
    prompt = read_prompt(cfg.prompt_file)
    negative_prompt = read_prompt(cfg.negative_prompt_file)
    print(f"The prompt is \n{prompt}")
    print(f"The negative prompt is \n{negative_prompt}")
    generator = Text2ImageGenerator(prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    model_name=cfg.model,
                                    batch_size=cfg.batch_size,
                                    scheduler_name=cfg.scheduler,
                                    num_inference_steps=cfg.num_inference_steps,
                                    random_seed=cfg.random_seed,
                                    height=cfg.height,
                                    width=cfg.width,
                                    guidance_scale=cfg.guidance_scale,
                                    device=device)
    generator.__call__()
