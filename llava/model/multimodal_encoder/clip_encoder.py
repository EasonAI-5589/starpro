"""CLIP vision tower with the text embeddings required by STAR-Pro QR."""

import torch
import torch.nn as nn
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizerFast,
    CLIPVisionConfig,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.text_tower = None

        if not delay_load or getattr(args, "unfreeze_mm_vision_tower", False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name, device_map=device_map
        )
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    @staticmethod
    def _resolve_device(device_map):
        if isinstance(device_map, dict):
            device_map = device_map.get("", "cuda")
        if isinstance(device_map, int):
            return torch.device(f"cuda:{device_map}")
        if isinstance(device_map, str) and device_map != "auto":
            return torch.device(device_map)
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def load_text_tower(self, device_map=None):
        if self.text_tower is not None:
            return
        device = self._resolve_device(device_map)
        projected_vision = CLIPVisionModelWithProjection.from_pretrained(
            self.vision_tower_name
        ).to(device)
        self.vision_tower.visual_projection = projected_vision.visual_projection
        self.text_tokenizer = CLIPTokenizerFast.from_pretrained(self.vision_tower_name)
        self.text_tower = CLIPTextModelWithProjection.from_pretrained(
            self.vision_tower_name
        ).to(device)
        self.text_tower.requires_grad_(False)
        self.max_position_embeddings = self.text_tower.config.max_position_embeddings

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            return image_features[:, 1:]
        if self.select_feature == "cls_patch":
            return image_features
        raise ValueError(f"unexpected select feature: {self.select_feature}")

    def _encode_text(self, texts):
        if self.text_tower is None:
            raise RuntimeError("STAR-Pro requires load_text_tower() before inference")
        text_inputs = self.text_tokenizer(text=texts, return_tensors="pt")
        input_length = text_inputs.input_ids.shape[1]
        segments = (input_length - 1) // self.max_position_embeddings + 1
        padding = segments * self.max_position_embeddings - input_length
        device = next(self.text_tower.parameters()).device
        text_inputs = {
            key: torch.cat([value, value.new_zeros((value.shape[0], padding))], dim=1)
            .reshape(-1, self.max_position_embeddings)
            .to(device)
            for key, value in text_inputs.items()
        }
        return self.text_tower(**text_inputs).text_embeds

    @torch.no_grad()
    def forward(self, images, texts=None, output_attentions=False):
        if isinstance(images, list):
            if texts is not None or output_attentions:
                raise ValueError("STAR-Pro expects a batched image tensor")
            return [
                self.feature_select(
                    self.vision_tower(
                        image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                        output_hidden_states=True,
                    )
                ).to(image.dtype)
                for image in images
            ]

        outputs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
            output_attentions=output_attentions,
        )
        image_features = self.feature_select(outputs).to(images.dtype)
        if output_attentions:
            attention = outputs.attentions[self.select_layer]
            if self.select_feature == "patch":
                attention = attention[:, :, 0, 1:]
            return image_features, attention.to(images.dtype)
        if texts is None:
            return image_features

        post_layernorm = self.vision_tower.vision_model.post_layernorm
        image_embeds = post_layernorm(
            self.feature_select(outputs).to(next(post_layernorm.parameters()).device)
        )
        projection = self.vision_tower.visual_projection
        image_embeds = projection(
            image_embeds.to(
                device=next(projection.parameters()).device,
                dtype=next(projection.parameters()).dtype,
            )
        )
        return image_features, image_embeds, self._encode_text(texts)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2
