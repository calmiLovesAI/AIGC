from diffusers import DiffusionPipeline, EulerDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler

from tools.data.image import save_image, save_ai_generated_image
from tools.platform.device import get_device, set_seed_based_on_device


def get_scheduler(scheduler_name: str):
    match scheduler_name.lower():
        case 'pndm':
            return PNDMScheduler
        case 'euler_discrete':
            return EulerDiscreteScheduler
        case 'dpm_solver_multistep':
            return DPMSolverMultistepScheduler
        case _:
            raise ValueError(f"Unsupported scheduler name {scheduler_name}")


def get_stable_diffusion_v1_5_output(cfg, prompt, scheduler_name='pndm', num_inference_steps=50):
    device = get_device(type=cfg['device'])
    pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',
                                                 use_safetensors=True)

    scheduler = get_scheduler(scheduler_name)
    pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)

    pipeline = pipeline.to(device)
    # Set the seed of the random number generator
    generator = set_seed_based_on_device(cfg['device'])

    img = pipeline(prompt, generator=generator, num_inference_steps=num_inference_steps).images[0]
    # save_image(img, save_folder="./test_samples/diffusion/")
    save_ai_generated_image(img, save_folder="./test_samples/diffusion/", prompt=prompt)
