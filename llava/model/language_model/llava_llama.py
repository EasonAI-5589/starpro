#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");

"""LLaVA entry point for STAR-Pro and the released comparison baselines."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modelling_llama_star import STARVLMModel
from .modeling_llama_fastv import FastVLlamaModel
from .modeling_llama_sparsevlm import SparseLlamaModel
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


BASELINE_METHODS = frozenset(("divprune", "cdp3", "fastv", "sparsevlm"))


def baseline_budget(config, method, total_tokens):
    """Translate a public total nominal budget into a per-crop schedule key."""
    aspect = getattr(config, "image_aspect_ratio", None)
    if aspect not in ("pad", "anyres"):
        raise ValueError("Baseline configurations require image_aspect_ratio=pad or anyres")
    if config.num_hidden_layers not in (32, 40):
        raise ValueError("Released baselines require a 32-layer or 40-layer Llama")
    crops = 5 if aspect == "anyres" else 1
    allowed = (64, 128) if method in ("fastv", "sparsevlm") else (32, 64, 128)
    totals = tuple(value * crops for value in allowed)
    if isinstance(total_tokens, bool) or not isinstance(total_tokens, int) or total_tokens not in totals:
        raise ValueError(f"{method} requires total nominal T in {totals}; got {total_tokens!r}")
    return crops, total_tokens // crops


class LlavaLlamaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)


class STARLlavaLlamaModel(LlavaMetaModel, STARVLMModel):
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig, star_config: dict):
        super().__init__(config, starvlm_config=star_config)


class FastVLlavaLlamaModel(LlavaMetaModel, FastVLlamaModel):
    config_class = LlavaLlamaConfig

    def __init__(self, config, fastv_config):
        super().__init__(config, fastv_config=fastv_config)


class SparseLlavaLlamaModel(LlavaMetaModel, SparseLlamaModel):
    config_class = LlavaLlamaConfig

    def __init__(self, config, sparsevlm_config):
        super().__init__(config, sparsevlm_config=sparsevlm_config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    """Vanilla LLaVA, STAR-Pro, DivPrune, CDPruner, FastV and SparseVLM."""

    config_class = LlavaLlamaConfig

    def __init__(
        self,
        config,
        pruning_method=None,
        visual_token_num=None,
        use_star=False,
        star_config=None,
        use_fastv=False,
        fastv_config=None,
        use_sparsevlm=False,
        sparsevlm_config=None,
        **kwargs,
    ):
        del kwargs
        super(LlamaForCausalLM, self).__init__(config)

        if pruning_method not in (None, "vanilla", "star_pro", *BASELINE_METHODS):
            raise ValueError(
                "Supported pruning methods: vanilla, star_pro, divprune, cdp3, fastv, sparsevlm; "
                f"got {pruning_method!r}"
            )
        if pruning_method in BASELINE_METHODS:
            if use_star or (use_fastv and pruning_method != "fastv") or (use_sparsevlm and pruning_method != "sparsevlm"):
                raise ValueError("Conflicting pruning method and decoder flags")
            self.public_baseline_crop_count, per_crop = baseline_budget(
                config, pruning_method, visual_token_num
            )
            self.target_visual_tokens = visual_token_num
        if use_star or pruning_method == "star_pro":
            if not star_config:
                raise ValueError("star_config is required for pruning_method='star_pro'")
            self.model = STARLlavaLlamaModel(config, star_config=star_config)
        elif pruning_method == "fastv":
            settings = {"K": 2, **(fastv_config or {}), "T": per_crop}
            self.model = FastVLlavaLlamaModel(config, fastv_config=settings)
        elif pruning_method == "sparsevlm":
            settings = {**(sparsevlm_config or {}), "T": per_crop}
            self.model = SparseLlavaLlamaModel(config, sparsevlm_config=settings)
        else:
            self.model = LlavaLlamaModel(config)

        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pruning_method = pruning_method or "vanilla"
        self.visual_token_num = per_crop if pruning_method in BASELINE_METHODS else visual_token_num

        # The bundled evaluation entry points expose optional efficiency metrics.
        self.start_latency = False
        self.phase = "prefill"
        self.start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        self.end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        self.prefill_latency = 0.0
        self.decode_latency = 0.0
        self.track_flops = False
        self.layer_visual_tokens = []
        self.total_flops = 0.0
        self.flops_count = 0
        self.post_init()

    def get_pruning_method(self):
        return self.pruning_method

    def get_visual_token_num(self):
        return self.visual_token_num

    def get_model(self):
        return self.model

    def calculate_flops(self):
        """Return decoder FLOPs for the recorded visual-token trajectory."""
        if not self.layer_visual_tokens:
            return 0.0
        d = self.config.hidden_size
        m = self.config.intermediate_size
        flops = sum(
            8 * n * d * d + 4 * n * n * d + 6 * n * d * m
            for n in self.layer_visual_tokens
        )
        return flops / 1e12

    def record_layer_tokens(self, layer_tokens):
        self.layer_visual_tokens = list(layer_tokens)
        if self.track_flops:
            self.total_flops += self.calculate_flops()
            self.flops_count += 1

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                _visual_token_num,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes=image_sizes,
            )

        if self.start_latency:
            if self.start_event is None:
                raise RuntimeError("CUDA latency measurements require an NVIDIA GPU")
            if self.phase == "prefill" and input_ids is None:
                self.start_event.record()
            elif self.phase == "decode" and input_ids is not None and input_ids.shape[1] == 1:
                self.start_event.record()

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.start_latency:
            if self.phase == "prefill" and input_ids is None:
                self.end_event.record()
                torch.cuda.synchronize()
                self.prefill_latency += self.start_event.elapsed_time(self.end_event)
                self.phase = "decode"
            elif self.phase == "decode" and input_ids is not None and input_ids.shape[1] == 1:
                self.end_event.record()
                torch.cuda.synchronize()
                self.decode_latency += self.start_event.elapsed_time(self.end_event)
                self.phase = "prefill"
        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        texts: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("inputs_embeds is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                visual_token_num,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                texts=texts,
            )
        else:
            if self.pruning_method in BASELINE_METHODS:
                raise ValueError("Released baseline generation requires one image")
            inputs_embeds = self.get_model().embed_tokens(inputs)
            visual_token_num = 0

        output = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if getattr(self.model, "layer_visual_tokens", None):
            self.record_layer_tokens(self.model.layer_visual_tokens)
            if self.pruning_method in BASELINE_METHODS:
                visual_token_num = self.model.visual_token_num
        return output, visual_token_num

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        texts = kwargs.pop("texts", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if texts is not None:
            inputs["texts"] = texts
        return inputs


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModelForCausalLM.register(LlavaLlamaConfig, LlavaLlamaForCausalLM)
