from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    # Check if MustDrop Vision Tower should be used
    # Reference: MustDrop clip_encoder.py - CLIPVisionTowerMustDrop
    use_mustdrop = getattr(vision_tower_cfg, 'use_mustdrop', False)

    if vision_tower.startswith("openai") or vision_tower.startswith("laion") or "clip" in vision_tower or "ShareGPT4V" in vision_tower:
        if use_mustdrop:
            from .clip_encoder_mustdrop import CLIPVisionTowerMustDrop
            return CLIPVisionTowerMustDrop(vision_tower, args=vision_tower_cfg, **kwargs)
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
