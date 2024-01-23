import os
import torch
from compel import Compel, ReturnedEmbeddingsType

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

from src.diffusion.lora import add_multiple_loras
from src.diffusion.prompt import text_embeddings
from src.utils.file_ops import get_absolute_path


def build_stable_diffusion_pipeline(pretrained_model, loras, prompts, negative_prompts, use_lora=False,
                                    requires_safety_checker=True, device='cuda', use_reimplemented_lpw=False, clip_skip=2):
    """
    Build a Stable Diffusion 1.5 pipeline
    :param pretrained_model: str or os.PathLike, A link to the .ckpt file on the hub or a path to a file containing all pipeline weights.
    :param loras: dict, lora model cfg
    :param prompts: List of str
    :param negative_prompts: List of str
    :param use_lora: bool, whether to load lora weights.
    :param requires_safety_checker: bool
    :param device:
    :param use_reimplemented_lpw: bool
    :return:
    """
    if os.path.isfile(pretrained_model):
        pretrained_model = get_absolute_path(pretrained_model)
        pipeline = StableDiffusionPipeline.from_single_file(pretrained_model,
                                                            use_safetensors=True,
                                                            load_safety_checker=requires_safety_checker).to(device)
    else:
        # from hugging face
        pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=pretrained_model,
                                                           use_safetensors=True,
                                                           requires_safety_checker=requires_safety_checker).to(device)
    try:
        pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
    except Exception as e:
        print(e)

    if use_lora:
        add_multiple_loras(pipeline, loras)

    # enable sliced attention computation.
    pipeline.enable_attention_slicing()
    # enable xformers
    pipeline.enable_xformers_memory_efficient_attention()

    # prompt weighting
    if use_reimplemented_lpw:
        prompt_embeddings, negative_prompt_embeddings = text_embeddings(pipeline, prompts, negative_prompts, clip_stop_at_last_layers=clip_skip)
    else:
        prompt_embeddings, negative_prompt_embeddings = compel_prompt_weighting_for_sd(pipeline, prompts, negative_prompts)

    return {
        'pipeline': pipeline,
        'prompt_embeddings': prompt_embeddings,
        'negative_prompt_embeddings': negative_prompt_embeddings
    }


def compel_prompt_weighting_for_sd(pipeline, prompts, negative_prompts):
    """
    Prepare the prompt-weighted embeddings with Compel for Stable Diffusion Model.
    :param pipeline:
    :param prompts: List of str
    :param negative_prompts: List of str
    :return:
    """
    compel_proc = Compel(tokenizer=pipeline.tokenizer,
                         text_encoder=pipeline.text_encoder,
                         truncate_long_prompts=False)

    with torch.no_grad():
        conditioning = compel_proc(prompts)
        negative_conditioning = compel_proc(negative_prompts)
        [prompt_embeddings,
         negative_prompt_embeddings] = compel_proc.pad_conditioning_tensors_to_same_length(
            [conditioning, negative_conditioning])
    return prompt_embeddings, negative_prompt_embeddings


def compel_prompt_weighting_for_sdxl(pipeline, prompts, negative_prompts):
    """
    Prepare the prompt-weighted embeddings with Compel for Stable Diffusion XL model.
    :param pipeline:
    :param prompts: List of str
    :param negative_prompts: List of str
    :return:
    """
    compel_proc = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                         text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                         truncate_long_prompts=False,
                         returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                         requires_pooled=[False, True])

    with torch.no_grad():
        conditioning, pooled = compel_proc(prompts)
        negative_conditioning, neg_pooled = compel_proc(negative_prompts)
        [prompt_embeddings, negative_prompt_embeddings] = compel_proc.pad_conditioning_tensors_to_same_length(
            [conditioning, negative_conditioning])
    return prompt_embeddings, pooled, negative_prompt_embeddings, neg_pooled


def build_stable_diffusion_xl_pipeline(pretrained_model, loras, prompts, negative_prompts, use_lora=False,
                                       requires_safety_checker=True, device='cuda'):
    """
    Build a Stable Diffusion XL pipeline
    :param pretrained_model: str or os.PathLike, A link to the .ckpt file on the hub or a path to a file containing all pipeline weights.
    :param loras: dict, lora model cfg
    :param prompts: List of str
    :param negative_prompts: List of str
    :param use_lora: bool, whether to load lora weights.
    :param requires_safety_checker: bool
    :param device:
    :return:
    """
    if os.path.isfile(pretrained_model):
        pretrained_model = get_absolute_path(pretrained_model)
        pipeline = StableDiffusionXLPipeline.from_single_file(pretrained_model,
                                                              use_safetensors=True,
                                                              load_safety_checker=requires_safety_checker).to(device)
    else:
        # from hugging face
        pipeline = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path=pretrained_model,
                                                             use_safetensors=True,
                                                             requires_safety_checker=requires_safety_checker).to(device)
    try:
        pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
    except Exception as e:
        print(e)

    if use_lora:
        add_multiple_loras(pipeline, loras)

    # enable sliced attention computation.
    pipeline.enable_attention_slicing()
    # enable xformers
    pipeline.enable_xformers_memory_efficient_attention()

    # prompt weighting
    prompt_embeddings, pooled, negative_prompt_embeddings, neg_pooled = compel_prompt_weighting_for_sdxl(pipeline,
                                                                                                         prompts,
                                                                                                         negative_prompts)

    return {
        'pipeline': pipeline,
        'prompt_embeddings': prompt_embeddings,
        'pooled': pooled,
        'negative_prompt_embeddings': negative_prompt_embeddings,
        'neg_pooled': neg_pooled
    }
