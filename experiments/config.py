project_cfg = {
    'device': 'gpu',
}

txt2img_cfg = {
    'prompt_file': "experiments/prompt.txt",
    'scheduler': "pndm",
    'num_inference_steps': 50,
    'batch_size': 1,
    'model': "runwayml/stable-diffusion-v1-5",
    'random_seed': -1,
    'height': 512,
    'width': 512,
    'guidance_scale': 7.5,
    'clip_skip': 2,
    'lora': {
        'enable': False,
        'civitai': False,
        'model': [],
        'weights': [],
        'scales': [],
        'location': 'both',  # 'unet' or 'both' (both unet and text encoder)
    }
}
