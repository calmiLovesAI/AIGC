# AIGC
Artificial Intelligence Generated Content.


# Installation
1. set the project root directory as `PYTHONPATH` environment variable.
2. set `HF_HOME` environment variable.
3. install requirements
```commandline
git clone https://github.com/calmiLovesAI/AIGC.git
cd AIGC
pip install -r requirements.txt
```
4. install huggingface-diffusers
```commandline
pip install --upgrade diffusers[torch]
```

# Text to Image
1. Create a new `txt2img.yaml` file in the experiments folder in the root directory, and specify the configuration items in it. 
2. Create two `.txt` files to store prompt words and negative prompt words and modify the value of `prompt_file` and `negative_prompt_file` in `txt2img.yaml`.
3. Run the following in the root directory
```commandline
python .\scripts\text2image.py -c .\experiments\txt2img.yaml
```