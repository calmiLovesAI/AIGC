# AIGC
Artificial Intelligence Generated Content.


# Installation
1. Set the project root directory as `PYTHONPATH` environment variable.
2. Set `HF_HOME` environment variable.
3. Install requirements
```commandline
git clone https://github.com/calmiLovesAI/AIGC.git
cd AIGC
pip install -r requirements.txt
```

4. Install huggingface-diffusers
```commandline
pip install --upgrade diffusers[torch]
```

5. Download [Real-ESRGAN](https://github.com/ai-forever/Real-ESRGAN)'s weights from the [HuggingFace page](https://huggingface.co/ai-forever/Real-ESRGAN)

# Text to Image
## Start with yaml config file
1. Create a new `txt2img.yaml` file in the experiments folder in the root directory, and specify the configuration items in it. 
2. Create two `.txt` files to store prompt words and negative prompt words and modify the value of `prompt_file` and `negative_prompt_file` in `txt2img.yaml`.
3. Run the following in the root directory
```commandline
python .\scripts\text2image.py -c .\experiments\txt2img.yaml
```
## Start with civitai generation data file
```commandline
python .\scripts\text2image.py -c civitai
```

# CV tasks:
## 1. command
```commandline
python .\app.py -t det2d -m [model_name] -i [model_id] -f [picture_dir]
```

### 2. Supported models
- [YoloS](https://huggingface.co/docs/transformers/model_doc/yolos)
- [DETR](https://huggingface.co/docs/transformers/model_doc/detr)
- [Deformable DETR](https://huggingface.co/docs/transformers/model_doc/deformable_detr)
- [Conditional DETR](https://huggingface.co/docs/transformers/model_doc/conditional_detr)