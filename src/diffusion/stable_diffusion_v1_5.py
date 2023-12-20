import torch
from diffusers import DiffusionPipeline

from src.diffusion.schedulers import diffusion_schedulers
from tools.data.image import save_ai_generated_image
from tools.platform.device import get_device, set_seed_based_on_device


def get_stable_diffusion_v1_5_output(prompt, scheduler_name='pndm', num_inference_steps=50, device=torch.device('cpu')):
    pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',
                                                 use_safetensors=True)

    # set scheduler
    scheduler = diffusion_schedulers[scheduler_name]
    pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)

    pipeline = pipeline.to(device)
    # Set the seed of the random number generator
    generator = set_seed_based_on_device(device)

    img = pipeline(prompt, generator=generator, num_inference_steps=num_inference_steps).images[0]
    # save_image(img, save_folder="./test_samples/diffusion/")
    save_ai_generated_image(img, save_folder="./test_samples/diffusion/", prompt=prompt)
