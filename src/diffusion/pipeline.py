import torch
import random

from compel import Compel

from src.diffusion.lora import add_multiple_loras
from src.diffusion.stable_diffusion import get_diffusion_model_ckpt, build_stable_diffusion_model
from src.diffusion.scheduler import diffusion_schedulers
from tools.data.image import save_ai_generated_image


class Text2ImagePipeline:
    def __init__(self,
                 prompt,
                 negative_prompt,
                 model_name,
                 loras,
                 lora_location,
                 batch_size=1,
                 scheduler_name='pndm',
                 num_inference_steps=50,
                 random_seed=-1,
                 height=512,
                 width=512,
                 guidance_scale=7.5,
                 use_fp16=False,
                 use_lora=True,
                 requires_safety_checker=True,
                 device=torch.device('cuda')):
        """
        Generate an image from a text description.
        :param prompt:
        :param negative_prompt: str, The prompt or prompts to guide what to not include in image generation. Ignored when not using guidance (guidance_scale < 1)
        :param model_name:
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
        :param use_fp16:
        :param requires_safety_checker: bool, default True.
        :param device:
        """
        self.prompts = batch_size * [prompt]
        self.negative_prompts = batch_size * [negative_prompt]
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.loras = loras
        pretrained_model = get_diffusion_model_ckpt(model_name)

        # initialize the pipeline
        self.pipeline = build_stable_diffusion_model(pretrained_model, device, requires_safety_checker)

        # add lora
        if use_lora:
            add_multiple_loras(self.pipeline, self.loras)

        # set scheduler
        self.scheduler = diffusion_schedulers[scheduler_name]
        self.pipeline.scheduler = self.scheduler.from_config(self.pipeline.scheduler.config)

        # enable sliced attention computation.
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_xformers_memory_efficient_attention()

        # A torch.Generator object enables reproducibility in a pipeline by setting a manual seed.
        self.generator, self.random_seeds = get_torch_generator(self.batch_size, random_seed=random_seed)

        # prompt weighting
        compel_proc = Compel(tokenizer=self.pipeline.tokenizer,
                             text_encoder=self.pipeline.text_encoder,
                             truncate_long_prompts=False)

        self.prompt_embeddings = compel_proc(self.prompts)
        self.negative_prompt_embeddings = compel_proc(self.negative_prompts)

    def __call__(self, *args, **kwargs):
        output_images = self.pipeline(prompt_embeds=self.prompt_embeddings,
                                      negative_prompt_embeds=self.negative_prompt_embeddings,
                                      generator=self.generator,
                                      num_inference_steps=self.num_inference_steps,
                                      height=self.height,
                                      width=self.width,
                                      guidance_scale=self.guidance_scale).images
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
            random_seeds = random_seed
            torch_generator = [torch.Generator("cuda").manual_seed(random_seed[i]) for i in range(batch_size)]

    return torch_generator[0] if len(torch_generator) == 1 else torch_generator, random_seeds
