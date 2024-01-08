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
4. Install [Real-ESRGAN](https://github.com/ai-forever/Real-ESRGAN) as a upscaler, also download the model weights from the [HuggingFace page](https://huggingface.co/ai-forever/Real-ESRGAN)
```commandline
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
```
5. Install huggingface-diffusers
```commandline
pip install --upgrade diffusers[torch]
```
6. Download the pretrained model.

# Text to Image
## Start with terminal
1. Create a new `txt2img.yaml` file in the experiments folder in the root directory, and specify the configuration items in it. 
2. Create two `.txt` files to store prompt words and negative prompt words and modify the value of `prompt_file` and `negative_prompt_file` in `txt2img.yaml`.
3. Run the following in the root directory
```commandline
python .\scripts\text2image.py -c .\experiments\txt2img.yaml
```

## Start with WebUI
```commandline
python .\ui\app.py
```