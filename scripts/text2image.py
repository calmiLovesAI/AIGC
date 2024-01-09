from experiments.config import project_cfg, txt2img_cfg
from src.diffusion.lora import preprocess_lora_cfg
from src.diffusion.prompt import read_prompt_from_file
from src.diffusion.txt2img_pipeline import Text2ImagePipeline
from tools.config_parser import load_task_cfg
from tools.platform.device import get_device

SUPPORT_MODELS = ['Stable Diffusion 1.5', 'Stable Diffusion XL']

if __name__ == '__main__':
    cfg = load_task_cfg(project_cfg, txt2img_cfg)
    device = get_device(cfg.device)

    print("===========SUPPORTED MODELS==============")
    for i, model_name in enumerate(SUPPORT_MODELS):
        print(f"{i}: {model_name}")
    user_input_model_index = int(input('Please input the index of text2image model: '))

    prompt = read_prompt_from_file(cfg.prompt_file)
    negative_prompt = read_prompt_from_file(cfg.negative_prompt_file)
    print(f"The prompt is \n{prompt}")
    print(f"The negative prompt is \n{negative_prompt}")
    loras = preprocess_lora_cfg(cfg.lora.model, cfg.lora.weights, cfg.lora.scales, from_civitai=cfg.lora.civitai)
    pipeline = Text2ImagePipeline(prompt=prompt,
                                  negative_prompt=negative_prompt,
                                  model_name=cfg.model,
                                  model_type=SUPPORT_MODELS[user_input_model_index],
                                  loras=loras,
                                  lora_location=cfg.lora.location,
                                  upscaler=cfg.upscaler,
                                  scale_factor=cfg.scale_factor,
                                  batch_size=cfg.batch_size,
                                  scheduler_name=cfg.scheduler,
                                  num_inference_steps=cfg.num_inference_steps,
                                  random_seed=cfg.random_seed,
                                  height=cfg.height,
                                  width=cfg.width,
                                  guidance_scale=cfg.guidance_scale,
                                  clip_skip=cfg.clip_skip,
                                  use_lora=cfg.lora.enable,
                                  requires_safety_checker=cfg.get('nsfw', True),
                                  device=device)
    pipeline.__call__()
