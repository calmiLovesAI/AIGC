project_cfg = dict(
    device="gpu",
)


txt2img_cfg = dict(
    prompt_file="experiments/prompt.txt",
    scheduler="pndm",
    num_inference_steps=50,
    batch_size=1,
    model="runwayml/stable-diffusion-v1-5",
    random_seed=30,
    height=768,
    width=512,
    guidance_scale=7.5,
    loras=[],
    lora_weight_names=[],
    lora_weight_scales=[],
    load_lora_into_unet_and_text_encoder=True,   # True for both, False for only unet
    use_lora=False,
)

if __name__ == '__main__':
    print(project_cfg)
