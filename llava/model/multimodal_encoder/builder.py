"""Vision tower construction for the LLaVA STAR-Pro overlay."""

import os

from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    if getattr(vision_tower_cfg, "s2", False):
        raise ValueError("The STAR-Pro overlay supports the standard CLIP vision tower only")
    vision_tower = getattr(
        vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None)
    )
    if isinstance(vision_tower, str) and (
        os.path.exists(vision_tower)
        or vision_tower.startswith("openai/")
        or vision_tower.startswith("laion/")
    ):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    raise ValueError(f"Unsupported vision tower: {vision_tower!r}")
