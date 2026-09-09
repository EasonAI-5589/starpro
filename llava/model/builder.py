#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");

"""Checkpoint loader for the released STAR-Pro and LLaVA baseline configurations."""

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
)
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    use_text_tower=False,
    **kwargs,
):
    """Load a full LLaVA-1.5/NeXT checkpoint and optional CLIP text tower."""
    if model_base is not None:
        raise ValueError(
            "This isolated release expects a full LLaVA checkpoint; "
            "model_base/projector-only and LoRA loading are outside the paper path."
        )
    if "llava" not in model_name.lower():
        raise ValueError(f"expected a LLaVA checkpoint, got model_name={model_name!r}")

    load_kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        load_kwargs["device_map"] = {"": device}
    elif device_map == "auto" and torch.cuda.device_count() == 1:
        load_kwargs["device_map"] = {"": 0}

    if load_8bit:
        load_kwargs["load_in_8bit"] = True
    elif load_4bit:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16
    if use_flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )

    if getattr(model.config, "mm_use_im_patch_token", True):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if getattr(model.config, "mm_use_im_start_end", False):
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if use_text_tower or kwargs.get("pruning_method") == "cdp3":
        vision_tower.load_text_tower(device_map=device_map)
    image_processor = vision_tower.image_processor
    context_len = getattr(model.config, "max_sequence_length", 2048)
    return tokenizer, model, image_processor, context_len
