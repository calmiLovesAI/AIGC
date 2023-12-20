project_cfg = dict(
    device="gpu",
)


txt2img_cfg = dict(
    prompt_file="experiments/prompt.txt",
    scheduler="pndm",
    num_inference_steps=50,
    batch_size=1,
    model="runwayml/stable-diffusion-v1-5",
)

if __name__ == '__main__':
    print(project_cfg)
