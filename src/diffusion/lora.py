import os.path

from src.diffusion.models import get_diffusion_model_ckpt
from tools.data.file_ops import get_absolute_path


class LoRA:
    def __init__(self, lora_model, lora_weight_name, lora_scale):
        """
        Low-Rank Adaptation (LoRA) is a popular training technique because it is fast and generates smaller file sizes.
        :param lora_model:
        :param lora_weight_name:
        :param lora_scale:  float, A value of 0 is the same as only using the base model weights,
                                   and a value of 1 is equivalent to using the fully finetuned LoRA.
        """
        self.model = lora_model
        self.weights = lora_weight_name
        self.scale = lora_scale


def parse_loras(loras, lora_weight_names, lora_scales):
    if not loras:
        return None
    assert len(loras) == len(lora_scales)
    if not lora_weight_names:
        lora_weight_names = [''] * len(loras)
    return [
        LoRA(lora_model=get_diffusion_model_ckpt(loras[i]),
             lora_weight_name=lora_weight_names[i],
             lora_scale=lora_scales[i])
        for i in range(len(loras))
    ]


def add_lora(pipeline, lora, location):
    params = {
        'pretrained_model_name_or_path_or_dict': lora.model,
    }
    if lora.weights:
        params.update({
            'weight_name': lora.weights,
        })
    if location == 'whole':
        pipeline.load_lora_weights(**params)
    else:
        pipeline.unet.load_attn_procs(**params)


def add_multiple_loras(pipeline, loras, location):
    if len(loras) == 1:
        add_lora(pipeline, lora=loras[0], location=location)
    for i in range(len(loras)):
        add_lora(pipeline, lora=loras[i], location=location)
    pipeline.fuse_lora()
