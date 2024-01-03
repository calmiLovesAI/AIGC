def preprocess_lora_cfg(lora_model, lora_weights, lora_scales):
    """
    Preprocess lora cfg dict, and get adapter names.
    :param lora_model: List of str,
    :param lora_weights: List of str,
    :param lora_scales: List of float,
    :return:
    """
    if len(lora_model) == len(lora_weights) == len(lora_scales):
        adapter_names = [f"adapter_{i}" for i in range(len(lora_model))]
        return {
            'lora_model': lora_model,
            'lora_weights': lora_weights,
            'lora_scales': lora_scales,
            'adapter_names': adapter_names
        }
    else:
        raise ValueError("Lengths of input arrays do not match")


def add_multiple_loras(pipeline, loras):
    n_loras = len(loras['lora_model'])
    for i in range(n_loras):
        weight_name = loras['lora_weights'][i]
        if not weight_name:
            pipeline.load_lora_weights(loras['lora_model'][i],
                                       adapter_name=loras['adapter_names'][i])
        else:
            pipeline.load_lora_weights(loras['lora_model'][i],
                                       weight_name=weight_name,
                                       adapter_name=loras['adapter_names'][i])
    pipeline.set_adapters(loras['adapter_names'], adapter_weights=loras['lora_scales'])

    # fuse adapters directly into the model weights.
    pipeline.fuse_lora()

    return pipeline
