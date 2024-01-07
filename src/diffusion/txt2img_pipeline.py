import torch
import random

from src.diffusion.stable_diffusion import build_stable_diffusion_pipeline, build_stable_diffusion_xl_pipeline
from src.diffusion.scheduler import diffusion_schedulers
from tools.data.image import save_ai_generated_image


class Text2ImagePipeline:
    def __init__(self,
                 prompt,
                 negative_prompt,
                 model_name,
                 model_type,
                 loras,
                 lora_location,
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

        # initialize the pipeline
        if self.model_type == 'SD':
            components = build_stable_diffusion_pipeline(model_name, loras, prompts=self.prompts,
                                                         negative_prompts=self.negative_prompts,
                                                         use_lora=use_lora,
                                                         requires_safety_checker=requires_safety_checker,
                                                         device=device)
            self.pipeline = components['pipeline']
            self.prompt_embeddings = components['prompt_embeddings']
            self.negative_prompt_embeddings = components['negative_prompt_embeddings']
        elif self.model_type == 'SDXL':
            components = build_stable_diffusion_xl_pipeline(model_name, loras, prompts=self.prompts,
                                                            negative_prompts=self.negative_prompts,
                                                            use_lora=use_lora,
                                                            requires_safety_checker=requires_safety_checker,
                                                            device=device)
            self.pipeline = components['pipeline']
            self.prompt_embeddings = components['prompt_embeddings']
            self.negative_prompt_embeddings = components['negative_prompt_embeddings']
            self.pooled_prompt_embeds = components['pooled']
            self.negative_pooled_prompt_embeds = components['neg_pooled']

        # set scheduler
        self._set_scheduler(scheduler_name)

        # A torch.Generator object enables reproducibility in a pipeline by setting a manual seed.
        self.generator, self.random_seeds = get_torch_generator(self.batch_size, random_seed=random_seed)

    def _set_scheduler(self, scheduler_name):
        try:
            self.scheduler = diffusion_schedulers[scheduler_name]
        except Exception:
            raise ValueError(f"The scheduler {scheduler_name} is not in diffusion_schedulers, "
                             f"only {diffusion_schedulers.keys()} are supported.")
        self.pipeline.scheduler = self.scheduler.from_config(self.pipeline.scheduler.config)

    def __call__(self, *args, **kwargs):
        params = {}
        if self.model_type == 'SDXL':
            params.update({'pooled_prompt_embeds': self.pooled_prompt_embeds,
                           'negative_pooled_prompt_embeds': self.negative_pooled_prompt_embeds})
        output_images = self.pipeline(prompt_embeds=self.prompt_embeddings,
                                      negative_prompt_embeds=self.negative_prompt_embeddings,
                                      generator=self.generator,
                                      num_inference_steps=self.num_inference_steps,
                                      height=self.height,
                                      width=self.width,
                                      guidance_scale=self.guidance_scale,
                                      clip_skip=self.clip_skip,
                                      **params).images
        for i, image in enumerate(output_images):
            save_ai_generated_image(image, seed=self.random_seeds[i], prompt=self.prompts[0])


def get_torch_generator(batch_size, random_seed):
    """
    Generate a list of PyTorch Generators based on given criteria.

    Arguments:
    - batch_size: The desired length of the list of torch.Generator instances.
    - random_seed: The criteria for generating torch.Generator instances.
                   If random_seed is -1 or [-1], generates a list of random torch.Generator instances.
                   Otherwise, generates torch.Generator instances based on the provided seed or list of seeds.

    Returns:
    - torch_generator: A list of torch.Generator instances based on the criteria provided.
    - random_seed: List of int.

    Note:
    - If random_seed is a single integer and batch_size is 1, a single torch.Generator instance is returned.
    """
    if random_seed == -1 or (isinstance(random_seed, list) and random_seed[0] == -1):
        random_seeds = [random.randint(1, 2 ** 32 - 1) for _ in range(batch_size)]
        torch_generator = [torch.Generator("cuda").manual_seed(r_seed) for r_seed in
                           random_seeds]
    else:
        if isinstance(random_seed, int):
            random_seeds = [random_seed]
            torch_generator = [torch.Generator("cuda").manual_seed(random_seed)]
        else:
            # random_seed is a list
            assert batch_size == len(random_seed)
            random_seeds = [random.randint(1, 2 ** 32 - 1) if n == -1 else n for n in random_seed]
            torch_generator = [torch.Generator("cuda").manual_seed(random_seeds[i]) for i in range(batch_size)]

    return torch_generator[0] if len(torch_generator) == 1 else torch_generator, random_seeds
