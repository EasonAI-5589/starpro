"""LLaVA-1.5 adapter for the STAR-Pro recovery tree.

This module deliberately lives outside VLMEvalKit.  The recovery tree changes
``model.generate`` to return both generated IDs and the actual visual-token
count; the stock VLMEvalKit adapter assumes the upstream LLaVA signature and
therefore cannot score the STAR-Pro pruning methods correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from vlmeval.vlm.base import BaseModel


@dataclass(frozen=True)
class _MustDropThreshold:
    global_thr: float
    individual_thr: float


_MUSTDROP_THRESHOLDS = {
    64: _MustDropThreshold(0.011, 0.01),
    128: _MustDropThreshold(0.0012, 0.001),
    192: _MustDropThreshold(0.001, 0.001),
}


class StarProLLaVA(BaseModel):
    """Run a recovered STAR-Pro LLaVA-1.5 model through VLMEvalKit datasets."""

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(
        self,
        model_path: str,
        pruning_method: str = "vanilla",
        visual_token_num: int = 576,
        max_new_tokens: int = 128,
        num_latent: int = 20,
        latent_pool_h: int = 5,
        latent_pool_w: int = 4,
        vscan_stage2_tokens: int = 32,
        vscan_prune_layer: int = 16,
    ) -> None:
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        self.pruning_method = pruning_method
        self.visual_token_num = visual_token_num
        self.last_visual_token_num = visual_token_num
        self.conv_mode = "llava_v1"

        use_fastv = pruning_method == "fastv"
        use_sparsevlm = pruning_method == "sparsevlm"
        use_d2p = pruning_method == "d2p"
        use_svdvlm = pruning_method == "svdvlm"
        use_prefixvlm = pruning_method == "prefixvlm"
        use_holov2 = pruning_method == "HoloV_2"
        use_idea = pruning_method == "Idea"
        use_prefixvlm_2 = pruning_method == "prefixvlm_2"
        use_pdrop = pruning_method == "pdrop"
        use_mustdrop = pruning_method == "mustdrop"
        use_star = pruning_method in {"star", "star_v2", "star_pro", "star_v5"}
        use_vscan = pruning_method == "vscan"

        threshold = _MUSTDROP_THRESHOLDS.get(
            visual_token_num, _MUSTDROP_THRESHOLDS[128]
        )
        mustdrop_config = {
            "pruning_layers": [2, 6, 10, 14],
            "global_thr": threshold.global_thr,
            "individual_thr": threshold.individual_thr,
            "keep_rate": 0.08,
            "merge_threshold": 0.8,
            "merge_window_size": (3, 3),
            "T": visual_token_num,
        }
        star_config = {
            "T": visual_token_num,
            "num_latent": num_latent,
            "latent_pool_size": (latent_pool_h, latent_pool_w),
            "mode": (
                "star_v5"
                if pruning_method == "star_v5"
                else "star_pro"
                if pruning_method == "star_pro"
                else "star_v2"
                if pruning_method == "star_v2"
                else "star"
            ),
            "debug": False,
        }
        vscan_config = {
            "stage1_tokens": visual_token_num,
            "stage2_tokens": vscan_stage2_tokens,
            "prune_layer": vscan_prune_layer,
        }
        use_text_tower = (
            pruning_method == "trim"
            or "cdp3" in pruning_method
            or "thcp" in pruning_method
            or pruning_method in {"star_pro", "prefixvlm"}
        )

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path,
                None,
                get_model_name_from_path(model_path),
                pruning_method=pruning_method,
                visual_token_num=visual_token_num,
                use_fastv=use_fastv,
                fastv_config={"K": 2, "T": visual_token_num},
                use_sparsevlm=use_sparsevlm,
                sparsevlm_config={"T": visual_token_num},
                use_d2p=use_d2p,
                d2p_config={"T": visual_token_num},
                use_svdvlm=use_svdvlm,
                svdvlm_config={"T": visual_token_num},
                use_prefixvlm=use_prefixvlm,
                prefixvlm_config={"T": visual_token_num},
                use_holov2=use_holov2,
                holov2_config={"T": visual_token_num},
                use_idea=use_idea,
                idea_config={"T": visual_token_num},
                use_prefixvlm_2=use_prefixvlm_2,
                prefixvlm_2_config={"T": visual_token_num},
                use_pdrop=use_pdrop,
                pdrop_config={"T": visual_token_num},
                use_star=use_star,
                star_config=star_config,
                use_mustdrop=use_mustdrop,
                mustdrop_config=mustdrop_config,
                use_vscan=use_vscan,
                vscan_config=vscan_config,
                use_text_tower=use_text_tower,
            )
        )
        # ``infer_data_job`` reads this field for its progress display. Vanilla
        # LLaVA does not define it, whereas several pruning implementations do.
        self._core_has_visual_token_num = hasattr(self.model.model, "visual_token_num")
        if not self._core_has_visual_token_num:
            self.model.model.visual_token_num = visual_token_num
        self._mustdrop = use_mustdrop
        self._mustdrop_config = mustdrop_config
        self._generation_kwargs = {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }

    @staticmethod
    def _content_and_images(message: list[dict[str, Any]]) -> tuple[str, list[str]]:
        content, images = [], []
        for item in message:
            if item["type"] == "text":
                content.append(item["value"])
            elif item["type"] == "image":
                images.append(item["value"])
        if len(images) != 1:
            raise ValueError(
                f"STAR-Pro LLaVA sidecar expects one image per sample, got {len(images)}"
            )
        return "\n".join(content), images

    def generate_inner(self, message: list[dict[str, Any]], dataset: str | None = None) -> str:
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        text_for_pruning, image_paths = self._content_and_images(message)
        question = text_for_pruning
        if self.model.config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(image_paths[0]).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device="cuda", non_blocking=True)

        with torch.inference_mode():
            if self._mustdrop:
                output_ids = self.model.generate(
                    self._mustdrop_config["global_thr"],
                    self._mustdrop_config["individual_thr"],
                    input_ids,
                    images=image_tensor,
                    image_sizes=(image.size,),
                    **self._generation_kwargs,
                )
                actual_vtn = getattr(self.model.model, "visual_token_num", self.visual_token_num)
            else:
                generated = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=(image.size,),
                    texts=text_for_pruning,
                    **self._generation_kwargs,
                )
                if isinstance(generated, tuple):
                    output_ids, actual_vtn = generated
                else:
                    output_ids = generated
                    actual_vtn = getattr(self.model.model, "visual_token_num", self.visual_token_num)

        self.last_visual_token_num = float(actual_vtn)
        if not self._core_has_visual_token_num:
            self.model.model.visual_token_num = self.last_visual_token_num
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
