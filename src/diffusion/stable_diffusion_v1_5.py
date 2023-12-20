import torch
from diffusers import DiffusionPipeline

from src.diffusion.schedulers import diffusion_schedulers
from tools.data.image import save_ai_generated_image


def get_stable_diffusion_v1_5_output(prompt, model_id, batch_size=1, scheduler_name='pndm', num_inference_steps=50,
                                     device=torch.device('cpu')):
    pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=model_id,
                                                 torch_dtype=torch.float16,
                                                 use_safetensors=True)

    # set scheduler
    scheduler = diffusion_schedulers[scheduler_name]
    pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)

    # enable sliced attention computation.
    pipeline.enable_attention_slicing()

    pipeline = pipeline.to(device)
    # Set the seed of the random number generator
    generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
    prompts = batch_size * [prompt]

    imgs = pipeline(prompt=prompts, generator=generator, num_inference_steps=num_inference_steps).images

    for img in imgs:
        save_ai_generated_image(img, save_folder="./test_samples/diffusion/", prompt=prompt)
