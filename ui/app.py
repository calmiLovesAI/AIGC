import os
import threading

import torch

import gradio as gr

from scripts.text2image import get_model_filenames
from src.diffusion.prompt import read_prompt_from_str
from src.diffusion.scheduler import get_scheduler_names
from src.diffusion.txt2img_pipeline import Text2ImagePipeline
from src.diffusion.upscaler import ALL_UPSCALERS
from tools.data.file_ops import get_absolute_path

STABLE_DIFFUSION_MODEL_ROOT = './downloads/stable_diffusion/'





def run_txt2img(model, model_type, prompt, negative_prompt, scheduler,
                num_inference_steps, batch_size, height, width,
                random_seed, guidance_scale, clip_skip, upscaler, scale_factor):
    prompt = read_prompt_from_str(prompt)
    negative_prompt = read_prompt_from_str(negative_prompt)
    print(f"The prompt is \n{prompt}")
    print(f"The negative prompt is \n{negative_prompt}")

    model = get_absolute_path(relative_path=STABLE_DIFFUSION_MODEL_ROOT + model)

    txt2img_pipeline = Text2ImagePipeline(prompt=prompt,
                                          negative_prompt=negative_prompt,
                                          model_name=model,
                                          model_type=model_type,
                                          loras={},
                                          lora_location='',
                                          batch_size=batch_size,
                                          scheduler_name=scheduler,
                                          num_inference_steps=num_inference_steps,
                                          random_seed=random_seed,
                                          height=height,
                                          width=width,
                                          guidance_scale=guidance_scale,
                                          clip_skip=clip_skip,
                                          upscaler=upscaler,
                                          scale_factor=scale_factor,
                                          use_lora=False,
                                          requires_safety_checker=False)
    outputs = txt2img_pipeline.__call__()

    return outputs


def main():
    with gr.Blocks() as ui:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    model = gr.Dropdown(label='Choose the model', choices=get_model_filenames())
                    model_type = gr.Dropdown(label='Choose the model type',
                                             choices=['Stable Diffusion 1.5', 'Stable Diffusion XL'])
                    scheduler = gr.Dropdown(label='Choose the scheduler', choices=get_scheduler_names())
                prompt = gr.Textbox(label='prompt', show_copy_button=True)
                negative_prompt = gr.Textbox(label='negative prompt', show_copy_button=True)
                with gr.Row():
                    num_inference_steps = gr.Slider(label='num of inference steps', minimum=10, maximum=100, value=30,
                                                    step=1)
                    batch_size = gr.Slider(label='batch size', minimum=1, maximum=64, value=1, step=1)
                with gr.Row():
                    height = gr.Number(label='height', value=768, precision=0, minimum=512, maximum=None, step=1)
                    width = gr.Number(label='width', value=512, precision=0, minimum=512, maximum=None, step=1)
                with gr.Row():
                    random_seed = gr.Number(label='random seed', value=-1, precision=0, minimum=-1)
                    guidance_scale = gr.Slider(label='guidance scale', value=7.0, minimum=1.0, maximum=20.0)
                    clip_skip = gr.Slider(label='clip skip', value=2, minimum=1, maximum=15, step=1)
                with gr.Row():
                    upscaler = gr.Dropdown(label='Choose the upscaler', choices=ALL_UPSCALERS)
                    scale_factor = gr.Slider(label='scale factor', minimum=2, maximum=8, value=2, step=2)

                run_btn = gr.Button(value='Generate')

            with gr.Column():
                output = gr.Gallery(height='60vh')

        run_btn.click(fn=run_txt2img,
                      inputs=[
                          model, model_type, prompt, negative_prompt, scheduler,
                          num_inference_steps, batch_size, height, width,
                          random_seed, guidance_scale, clip_skip, upscaler,
                          scale_factor
                      ],
                      outputs=output)
    ui.launch()


if __name__ == '__main__':
    main()
