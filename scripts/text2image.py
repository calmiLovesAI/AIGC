import os

from experiments.config import project_cfg, txt2img_cfg
from src.diffusion.lora import preprocess_lora_cfg
from src.diffusion.prompt import read_prompt_from_file
from src.diffusion.txt2img_pipeline import Text2ImagePipeline
from src.utils.config_parser import load_task_cfg
from src.utils.file_ops import get_absolute_path
from src.utils.device import get_device


def get_model_filenames(root_dir, suffix='safetensors'):
    """
    Get all file names with the specified suffix in the root directory and return them.
    :param root_dir:
    :param suffix:
    :return:
    """
    model_files = []
    files = os.listdir(get_absolute_path(relative_path=root_dir))

    for filename in files:
        if filename.endswith(suffix):
            model_files.append(filename)

    return model_files


def set_model_type(cfg):
    model_types = ['Stable Diffusion 1.5', 'Stable Diffusion XL']
    print(f"================SUPPORTED MODEL TYPES==============")
    for i, model_name in enumerate(model_types):
        print(f"{i}: {model_name}")
    user_input_model_index = input('Please input the index of text2image model type(n for ignore): ')
    if user_input_model_index != 'n':
        return model_types[int(user_input_model_index)]
    return cfg.model_type


def set_model(cfg):
    root = './downloads/stable_diffusion/'
    models = get_model_filenames(root)
    models.append('from_huggingface')
    print(f"==================SUPPORTED MODELS=================")
    for i, elem in enumerate(models):
        print(f"{i}: {elem}")
    user_input_model_index = input('Please input the index of text2image model(n for ignore): ')
    if user_input_model_index != 'n':
        if user_input_model_index in [-1, len(models) - 1]:
            return cfg.model
        else:
            return os.path.join(root, models[int(user_input_model_index)])
    return cfg.model


def get_prompt(cfg):
    prompt = read_prompt_from_file(cfg.prompt_file, True)
    negative_prompt = read_prompt_from_file(cfg.negative_prompt_file, True)
    print(f"\n==================PROMPT=================")
    print(prompt)
    print(f"\n=============NEGATIVE PROMPT=============")
    print(negative_prompt)
    print(f"\n=========================================")
    return prompt, negative_prompt


def main():
    cfg = load_task_cfg(project_cfg, txt2img_cfg)
    device = get_device(cfg.device)

    model_type = set_model_type(cfg)
    model = set_model(cfg)

    print(f"You have selected the {model_type} model: {model}")

    if cfg.from_civitai:
        prompt = cfg.prompt
        negative_prompt = cfg.negative_prompt
    else:
        prompt, negative_prompt = get_prompt(cfg)

    loras = preprocess_lora_cfg(cfg.lora.model, cfg.lora.weights, cfg.lora.scales, from_civitai=cfg.lora.civitai)
    pipeline = Text2ImagePipeline(prompt=prompt,
                                  negative_prompt=negative_prompt,
                                  model_name=model,
                                  model_type=model_type,
                                  loras=loras,
                                  lora_location=cfg.lora.location,
                                  upscaler=cfg.upscaler,
                                  output_path=cfg.output_root,
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
                                  use_lpw=cfg.get('lpw', True),
                                  requires_safety_checker=cfg.get('nsfw', True),
                                  device=device)
    pipeline.__call__()


if __name__ == '__main__':
    main()
