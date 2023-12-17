from diffusers import DiffusionPipeline, EulerDiscreteScheduler

from tools.data.image import save_image, save_ai_generated_image


def get_stable_diffusion_v1_5_output(prompt):
    pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',
                                                 use_safetensors=True)
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.to('cuda')

    img = pipeline(prompt).images[0]
    # save_image(img, save_folder="./test_samples/diffusion/")
    save_ai_generated_image(img, save_folder="./test_samples/diffusion/", prompt=prompt)
