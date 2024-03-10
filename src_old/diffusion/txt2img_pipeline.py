import torch
import random

from src_old.diffusion.long_prompt_weighting import LongPromptWeightingAdapter
from src_old.diffusion.stable_diffusion import build_stable_diffusion_pipeline, build_stable_diffusion_xl_pipeline
from src_old.diffusion.scheduler import get_scheduler
from src_old.diffusion.upscaler import upscale_image
from src_old.utils.image import save_ai_generated_image


class Text2ImagePipeline:
    def __init__(self,
                 prompt,
                 negative_prompt,
                 model_name,
                 model_type,
                 loras,
                 lora_location,
                 upscaler,
                 output_path,
                 scale_factor=2,
                 batch_size=1,
                 scheduler_name='pndm',
                 num_inference_steps=50,
                 random_seed=-1,
                 height=512,
                 width=512,
                 guidance_scale=7.5,
                 clip_skip=2,
                 use_fp16=False,
                 use_lora=True,
                 requires_safety_checker=True,
                 device=torch.device('cuda')):
        """
        Generate an image from a text description.
        :param prompt:
        :param negative_prompt: str, The prompt or prompts to guide what to not include in image generation. Ignored when not using guidance (guidance_scale < 1)
        :param model_name:
        :param model_type: str
        :param loras: dict, lora model cfg
        :param lora_location: str, 'whole' for loading LoRA weights into both the UNet and text encoder, 'unet' for only the UNet.
        :param upscaler:
        :param output_path:
        :param scale_factor:
        :param batch_size:
        :param scheduler_name:
        :param num_inference_steps:
        :param random_seed:
        :param height: int, must be a multiple of 8
        :param width: int, must be a multiple of 8
        :param guidance_scale: float, A higher guidance scale value encourages the model to generate images
                               closely linked to the text prompt at the expense of lower image quality.
                               Guidance scale is enabled when guidance_scale > 1.
        :param clip_skip: int, default 2. Number of layers to be skipped from CLIP while computing the prompt embeddings.
                          A value of 1 means that the output of the pre-final layer will be used for computing the prompt embeddings.
        :param use_fp16:
        :param use_lora:
        :param requires_safety_checker: bool, default True.
        :param device:
        """
        self.model_type = model_type
        self.prompts = batch_size * [prompt]
        self.negative_prompts = batch_size * [negative_prompt]
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.clip_skip = clip_skip
        self.loras = loras
        self.upscaler = upscaler
        self.scale_factor = scale_factor
        self.output_path = output_path

        # initialize the pipeline
        if self.model_type == 'Stable Diffusion 1.5':
            self.pipeline = build_stable_diffusion_pipeline(model_name, loras,
                                                            use_lora=use_lora,
                                                            requires_safety_checker=requires_safety_checker,
                                                            device=device)
        elif self.model_type == 'Stable Diffusion XL':
            self.pipeline = build_stable_diffusion_xl_pipeline(model_name, loras,
                                                               use_lora=use_lora,
                                                               requires_safety_checker=requires_safety_checker,
                                                               device=device)
        self.lpw_adapter = LongPromptWeightingAdapter(self.pipeline)
        # set scheduler
        self._set_scheduler(scheduler_name)

        # A torch.Generator object enables reproducibility in a pipeline by setting a manual seed.
        self.generator, self.random_seeds = get_torch_generator(self.batch_size, input_random_seed=random_seed)

    def _set_scheduler(self, scheduler_name):
        self.scheduler = get_scheduler(scheduler_name)
        self.pipeline.scheduler = self.scheduler.from_config(self.pipeline.scheduler.config)

    def __call__(self, *args, **kwargs):
        if self.model_type == 'Stable Diffusion 1.5':
            prompt_embeds, negative_prompt_embeds = self.lpw_adapter(self.prompts, self.negative_prompts, clip_skip=self.clip_skip)
            print(f"prompt_embedding: {prompt_embeds.shape}, neg_embedding: {negative_prompt_embeds}")
            output_images = self.pipeline(prompt_embeds=prompt_embeds,
                                          negative_prompt_embeds=negative_prompt_embeds,
                                          generator=self.generator,
                                          num_inference_steps=self.num_inference_steps,
                                          height=self.height,
                                          width=self.width,
                                          guidance_scale=self.guidance_scale).images
        else:
            output_images = self.pipeline(prompt=self.prompts,
                                          negative_prompt=self.negative_prompts,
                                          generator=self.generator,
                                          num_inference_steps=self.num_inference_steps,
                                          height=self.height,
                                          width=self.width,
                                          guidance_scale=self.guidance_scale,
                                          clip_skip=self.clip_skip).images
        if self.scale_factor > 1:
            output_images = upscale_image(images=output_images, model=self.upscaler, scale_factor=self.scale_factor)
        for i, image in enumerate(output_images):
            save_ai_generated_image(image, seed=self.random_seeds[i], save_folder=self.output_path,
                                    prompt=self.prompts[0])
        return output_images


def get_torch_generator(batch_size, input_random_seed):
    """
    Generate a list of PyTorch Generators based on given criteria.

    Arguments:
    - batch_size: The desired length of the list of torch.Generator instances.
    - input_random_seed: The criteria for generating torch.Generator instances.
                         If random_seed is -1 or [-1], generates a list of random torch.Generator instances.
                         Otherwise, generates torch.Generator instances based on the provided seed or list of seeds.

    Returns:
    - torch_generator: A list of torch.Generator instances based on the criteria provided.
    - random_seed: List of int.

    Note:
    - If random_seed is a single integer and batch_size is 1, a single torch.Generator instance is returned.
    """
    random_seeds = [random.randint(1, 2 ** 32 - 1) for _ in range(batch_size)]

    if input_random_seed == -1:
        pass
    else:
        if isinstance(input_random_seed, int):
            random_seeds[0] = input_random_seed
        else:
            # random_seed is a list
            for i in range(batch_size):
                if input_random_seed[i] != -1:
                    random_seeds[i] = input_random_seed[i]

    torch_generator = [torch.Generator("cuda").manual_seed(seed) for seed in random_seeds]
    return torch_generator, random_seeds
