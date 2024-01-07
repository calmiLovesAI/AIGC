import os

import gradio as gr

from src.diffusion.scheduler import get_scheduler_names
from src.diffusion.txt2img_pipeline import Text2ImagePipeline
from tools.data.file_ops import get_absolute_path

STABLE_DIFFUSION_MODEL_ROOT = './downloads/stable_diffusion/'


def txt2img(model, model_type, prompt, negative_prompt, scheduler,
            num_inference_steps, batch_size, height, width,
            random_seed, guidance_scale, clip_skip):
    model = get_absolute_path(relative_path=STABLE_DIFFUSION_MODEL_ROOT + model)
    ret = Text2ImagePipeline(prompt=prompt,
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
                             use_lora=False,
                             requires_safety_checker=False).__call__(show=True)
    if len(ret) < 2:
        return ret[0], None
    else:
        return ret[0], ret[1]


def get_model_filenames(root_dir=STABLE_DIFFUSION_MODEL_ROOT, suffix='safetensors'):
    """
    Get all file names with the specified suffix in the root directory and return them.
    :param root_dir:
    :param suffix:
    :return:
    """
    model_files = []
    files = os.listdir(get_absolute_path(relative_path=root_dir))

    for filename in files:
        if filename.endswith(suffix):
            model_files.append(filename)

    return model_files


def main():
    with gr.Blocks() as ui:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    model = gr.Dropdown(label='Choose the model', choices=get_model_filenames())
                    model_type = gr.Dropdown(label='Choose the model type', choices=['Stable Diffusion 1.5', 'Stable Diffusion XL'])
                    scheduler = gr.Dropdown(label='Choose the scheduler', choices=get_scheduler_names())
                prompt = gr.Textbox(label='prompt', interactive=True, show_copy_button=True)
                negative_prompt = gr.Textbox(label='negative prompt', show_copy_button=True)
                with gr.Row():
                    num_inference_steps = gr.Slider(label='num of inference steps', minimum=10, maximum=100, value=20, step=1)
                    batch_size = gr.Slider(label='batch size', minimum=1, maximum=64, value=1, step=1)
                with gr.Row():
                    height = gr.Number(label='height', value=1024, precision=0, minimum=512, maximum=None, step=1)
                    width = gr.Number(label='width', value=640, precision=0, minimum=512, maximum=None, step=1)
                with gr.Row():
                    random_seed = gr.Number(label='random seed', value=-1, precision=0, minimum=-1)
                    guidance_scale = gr.Number(label='guidance scale', value=7.0, minimum=1.0)
                    clip_skip = gr.Number(label='clip skip', value=2, precision=0)

                run_btn = gr.Button(value='Run')

            with gr.Column():
                with gr.Row():
                    output_1 = gr.Image(type='pil')
                    output_2 = gr.Image(type='pil')

        run_btn.click(fn=txt2img,
                      inputs=[
                          model, model_type, prompt, negative_prompt, scheduler,
                          num_inference_steps, batch_size, height, width,
                          random_seed, guidance_scale, clip_skip
                      ],
                      outputs=[
                          output_1,
                          output_2
                      ])
    ui.launch()


if __name__ == '__main__':
    main()
