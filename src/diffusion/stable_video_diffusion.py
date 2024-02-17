import os
from datetime import date
import random

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

from src.utils.file_ops import get_absolute_path, generate_random_filename, create_directory_if_not_exists


def generate_video_with_svd_xt(condition_image_path_or_url: str,
                               seed: int = -1,
                               height: int = 576,
                               width: int = 1024,
                               num_inference_steps: int = 25,
                               min_guidance_scale: float = 1.0,
                               max_guidance_scale: float = 3.0,
                               fps: int = 7,
                               num_frames: int = 25,
                               motion_bucket_id: int = 127,
                               noise_aug_strength: float = 0.02,
                               output_video_path: str = None,
                               device='cuda'):
    if seed == -1:
        seed = random.randint(1, 2 ** 32 - 1)
        generator = torch.manual_seed(seed)
    else:
        generator = torch.manual_seed(seed)

    if output_video_path is None:
        root_folder = get_absolute_path(relative_path='outputs')
        # Name of sub folder is the current date.
        sub_folder = date.today().isoformat()
        file_prefix = os.path.splitext(os.path.basename(condition_image_path_or_url))[0][:15]
        file_name = generate_random_filename(file_prefix=file_prefix, suffix=f"_seed={seed}.mp4")
        output_video_path = os.path.join(root_folder, sub_folder, file_name)
        create_directory_if_not_exists(directory_path=output_video_path, is_file=True)
    if os.path.isfile(condition_image_path_or_url):
        condition_image_path_or_url = get_absolute_path(condition_image_path_or_url)

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)
    try:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    except Exception as e:
        print(e)
    pipe.enable_model_cpu_offload()
    pipe.unet.enable_forward_chunking()
    image = load_image(condition_image_path_or_url)
    image = image.resize(size=(height, width))



    frames = pipe(image=image,
                  height=height,
                  width=width,
                  num_inference_steps=num_inference_steps,
                  min_guidance_scale=min_guidance_scale,
                  max_guidance_scale=max_guidance_scale,
                  decode_chunk_size=2,
                  generator=generator,
                  num_frames=num_frames,
                  motion_bucket_id=motion_bucket_id,
                  noise_aug_strength=noise_aug_strength,
                  fps=fps).frames[0]

    export_to_video(frames, output_video_path=output_video_path, fps=fps)
