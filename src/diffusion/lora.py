def preprocess_lora_cfg(lora_model, lora_weights, lora_scales, from_civitai=False):
    """
    Preprocess lora cfg dict, and get adapter names.
    :param lora_model: List of str,
    :param lora_weights: List of str,
    :param lora_scales: List of float,
    :param from_civitai: bool,
    :return:
    """
    if from_civitai:
        assert len(lora_weights) == len(lora_scales)
        lora_root = lora_model[0]
        lora_model = [lora_root] * len(lora_weights)
    else:
        assert len(lora_model) == len(lora_weights) == len(lora_scales)
    adapter_names = [f"adapter_{i}" for i in range(len(lora_model))]
    return {
        'lora_model': lora_model,
        'lora_weights': lora_weights,
        'lora_scales': lora_scales,
        'adapter_names': adapter_names
    }


def add_multiple_loras(pipeline, loras):
    n_loras = len(loras['lora_model'])
    for i in range(n_loras):
        pipeline.load_lora_weights(loras['lora_model'][i],
                                   weight_name=loras['lora_weights'][i],
                                   adapter_name=loras['adapter_names'][i])
    pipeline.set_adapters(loras['adapter_names'], adapter_weights=loras['lora_scales'])

    # fuse adapters directly into the model weights.
    pipeline.fuse_lora()

    return pipeline
