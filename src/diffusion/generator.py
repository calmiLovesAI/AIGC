import torch
from diffusers import DiffusionPipeline, AutoPipelineForText2Image

from src.diffusion.lora import add_multiple_loras
from src.diffusion.scheduler import diffusion_schedulers
from tools.data.image import save_ai_generated_image


class Text2ImageGenerator:
    def __init__(self, prompt,
                 negative_prompt,
                 model_name,
                 loras,
                 lora_mode,
                 batch_size=1,
                 scheduler_name='pndm',
                 num_inference_steps=50,
                 random_seed=0,
                 height=512,
                 width=512,
                 guidance_scale=7.5,
                 use_fp16=False,
                 device=torch.device('cuda')):
        """
        Generate an image from a text description.
        :param prompt:
        :param negative_prompt: str, The prompt or prompts to guide what to not include in image generation. Ignored when not using guidance (guidance_scale < 1)
        :param model_name:
        :param loras: List of LoRAs,
        :param lora_mode: bool, True for loading LoRA weights into both the UNet and text encoder, False for only the UNet.
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
        :param device:
        """
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.loras = loras

        # initialize the pipeline
        if use_fp16:
            self.pipeline = AutoPipelineForText2Image.from_pretrained(pretrained_model_or_path=model_name,
                                                                      torch_dtype=torch.float16,
                                                                      variant='fp16',
                                                                      use_safetensors=True).to(device)
        else:
            self.pipeline = AutoPipelineForText2Image.from_pretrained(pretrained_model_or_path=model_name,
                                                                      use_safetensors=True).to(device)

        try:
            self.pipeline.unet = torch.compile(self.pipeline.unet, mode="reduce-overhead", fullgraph=True)
        except Exception as e:
            print(e)

        # add lora
        add_multiple_loras(self.pipeline, self.loras, mode=lora_mode)

        # set scheduler
        self.scheduler = diffusion_schedulers[scheduler_name]
        self.pipeline.scheduler = self.scheduler.from_config(self.pipeline.scheduler.config)

        # enable sliced attention computation.
        self.pipeline.enable_attention_slicing()

        # A torch.Generator object enables reproducibility in a pipeline by setting a manual seed.
        if batch_size == 1:
            self.generator = torch.Generator("cuda").manual_seed(random_seed)
        else:
            self.generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
        self.prompts = batch_size * [prompt]

    def __call__(self, *args, **kwargs):
        call_parameters = {
            'prompt': self.prompts,
            'generator': self.generator,
            'num_inference_steps': self.num_inference_steps,
            'height': self.height,
            'width': self.width,
            'guidance_scale': self.guidance_scale
        }

        if self.negative_prompt:
            call_parameters.update({'negative_prompt': self.negative_prompt})
        if len(self.loras) == 1:
            call_parameters.update({' cross_attention_kwargs': {'scale': self.loras[0].scale}})
        output_images = self.pipeline(**call_parameters).images
        for image in output_images:
            save_ai_generated_image(image, prompt=self.prompt)
