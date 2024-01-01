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
    assert len(loras) == len(lora_scales)
    if not lora_weight_names:
        lora_weight_names = [''] * len(loras)
    return [
        LoRA(lora_model=loras[i],
             lora_weight_name=lora_weight_names[i],
             lora_scale=lora_scales[i])
        for i in range(len(loras))
    ]


def add_lora(pipeline, lora, mode):
    params = {
        'pretrained_model_name_or_path_or_dict': lora.model,
    }
    if lora.weights:
        params.update({
            'weight_name': lora.weights,
        })
    if mode:
        pipeline.load_lora_weights(**params)
    else:
        pipeline.unet.load_attn_procs(**params)


def add_multiple_loras(pipeline, loras, mode):
    if len(loras) == 1:
        add_lora(pipeline, lora=loras[0], mode=mode)
    for i in range(len(loras)):
        add_lora(pipeline, lora=loras[i], mode=mode)
    pipeline.fuse_lora()
