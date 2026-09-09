"""Tensor contracts for released baselines, without checkpoints or a GPU.

Run with the pinned requirements installed:
    python -m unittest discover -s tests -p 'test_baseline_models.py' -v
The ordinary stdlib-only release check skips these tensor tests.
"""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "einops"))

if HAS_RUNTIME:
    import torch
    from torch import nn


def load_model_module():
    """Load the real overlay; stub only upstream image-grid/projector helpers."""
    package_name = "release_baseline_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "llava" / "model")]
    language = types.ModuleType(package_name + ".language_model")
    language.__path__ = [str(ROOT / "llava" / "model" / "language_model")]
    constants = types.ModuleType("llava.constants")
    constants.IGNORE_INDEX = -100
    constants.IMAGE_TOKEN_INDEX = -200
    constants.DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    constants.DEFAULT_IM_START_TOKEN = "<im_start>"
    constants.DEFAULT_IM_END_TOKEN = "<im_end>"
    mm_utils = types.ModuleType("llava.mm_utils")
    mm_utils.get_anyres_image_grid_shape = lambda *args: (2, 2)
    projector = types.ModuleType(package_name + ".multimodal_projector.builder")
    projector.build_vision_projector = lambda config: nn.Identity()
    name = language.__name__ + ".llava_llama"
    spec = importlib.util.spec_from_file_location(name, Path(language.__path__[0]) / "llava_llama.py")
    module = importlib.util.module_from_spec(spec)
    # Leave the loaded local package registered so decoder relative imports
    # retain their identity; no checkpoints or upstream CLIP downloads occur.
    sys.modules[package_name] = package
    sys.modules[language.__name__] = language
    sys.modules[name] = module
    with mock.patch.dict(sys.modules, {
        "llava.constants": constants, "llava.mm_utils": mm_utils,
        projector.__name__: projector,
    }):
        spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_RUNTIME, "Install pinned torch/transformers/einops for tensor contracts")
class BaselineModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.module = load_model_module()

    def config(self, aspect="pad", layers=32):
        config = self.module.LlavaLlamaConfig(
            vocab_size=32, hidden_size=64, intermediate_size=96,
            num_hidden_layers=layers, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=4096,
            bos_token_id=1, eos_token_id=None, pad_token_id=0,
        )
        config.image_aspect_ratio = aspect
        config.mm_patch_merge_type = "spatial_unpad" if aspect == "anyres" else "flat"
        config.image_grid_pinpoints = [[672, 672]]
        config._attn_implementation = "eager"
        return config

    def model(self, method, total=64, aspect="pad", layers=32, attention="eager", max_positions=4096):
        torch.manual_seed(17)
        config = self.config(aspect, layers)
        config._attn_implementation = attention
        config.max_position_embeddings = max_positions
        model = self.module.LlavaLlamaForCausalLM(
            config, pruning_method=method, visual_token_num=total,
            fastv_config={"K": 2, "T": total}, sparsevlm_config={"T": total},
        ).eval()

        class FixtureVision(nn.Module):
            num_patches_per_side = 24
            config = types.SimpleNamespace(image_size=336)

            def __init__(self):
                super().__init__()
                self.register_buffer("features", torch.randn(1, 576, 64))
                self.register_buffer("image_embeds", torch.randn(1, 576, 64))
                self.register_buffer("text_embeds", torch.randn(1, 64))

            def forward(self, images, texts=None):
                features = self.features.expand(images.shape[0], -1, -1).clone()
                if texts is not None:
                    return features, self.image_embeds.expand(images.shape[0], -1, -1), self.text_embeds
                return features

        model.model.vision_tower = FixtureVision()
        model.model.mm_projector = nn.Identity()
        return model

    def pack(self, model, prefix=7, crops=1, masked_prefix=False):
        ids = torch.tensor([[7] * prefix + [-200, 9, 10, 11]])
        mask = torch.ones_like(ids)
        if masked_prefix:
            mask[0, 0] = 0
        images = torch.zeros(1, 3, 8, 8) if crops == 1 else [torch.zeros(crops, 3, 8, 8)]
        return model.prepare_inputs_labels_for_multimodal(
            ids, None, mask, None, None, images,
            image_sizes=[(672, 672)], texts="Which object is shown?",
        )

    def test_total_budget_translation_and_unsupported_budgets(self):
        for method in ("divprune", "cdp3", "fastv", "sparsevlm"):
            for aspect, crops in (("pad", 1), ("anyres", 5)):
                with self.subTest(method=method, aspect=aspect):
                    self.assertEqual(self.module.baseline_budget(self.config(aspect), method, 64 * crops), (crops, 64))
                    self.assertEqual(self.module.baseline_budget(self.config(aspect), method, 128 * crops), (crops, 128))
                    if method in ("fastv", "sparsevlm"):
                        with self.assertRaisesRegex(ValueError, "total nominal T"):
                            self.module.baseline_budget(self.config(aspect), method, 32 * crops)
        for method in ("fastv", "sparsevlm"):
            model = self.model(method, 320, "anyres")
            self.assertEqual(model.visual_token_num, 64)
            self.assertEqual(model.target_visual_tokens, 320)
            self.assertEqual(model.model.visual_token_length, 2880)
            if method == "fastv":
                self.assertEqual(model.model.R, 29 * 5)
            else:
                self.assertEqual(model.model.retained_num, 64)
        with self.assertRaisesRegex(ValueError, "image_aspect_ratio"):
            self.module.baseline_budget(self.config("square"), "divprune", 64)

    def test_predecoder_selectors_pack_exact_pad_and_five_crop_counts(self):
        for method in ("divprune", "cdp3"):
            for aspect, crops in (("pad", 1), ("anyres", 5)):
                with self.subTest(method=method, aspect=aspect):
                    model = self.model(method, 32 * crops, aspect)
                    packed = self.pack(model, crops=crops)
                    self.assertEqual(packed[6], 32 * crops)
                    self.assertEqual(packed[4].shape, (1, 7 + 32 * crops + 3, 64))
                    self.assertTrue(torch.isfinite(packed[4]).all())

    def test_measured_span_crop_batch_and_image_guards(self):
        model = self.model("fastv")
        self.pack(model, prefix=19, masked_prefix=True)
        self.assertEqual(model.model.system_prompt_length, 18)
        self.assertEqual(model.model.visual_token_length, 576)
        with self.assertRaisesRegex(ValueError, "crop"):
            self.pack(self.model("divprune", 160, "anyres"), crops=4)
        ids = torch.tensor([[1, -200, -200, 2]])
        with self.assertRaisesRegex(ValueError, "one image"):
            model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, torch.zeros(1, 3, 8, 8))
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            model.prepare_inputs_labels_for_multimodal(ids.repeat(2, 1), None, None, None, None, torch.zeros(2, 3, 8, 8))

    @torch.no_grad() if HAS_RUNTIME else (lambda fn: fn)
    def test_prefill_attention_masks_and_three_cached_decode_steps(self):
        for method in ("fastv", "sparsevlm"):
            for prefix in (3, 19):
                with self.subTest(method=method, prefix=prefix):
                    model = self.model(method)
                    packed = self.pack(model, prefix=prefix)
                    embeds, mask = packed[4], packed[2]
                    output = model.model(inputs_embeds=embeds, attention_mask=mask, output_attentions=True, use_cache=True)
                    original_length = prefix + 576 + 3
                    self.assertEqual(len(output.attentions), 32)
                    expected_trajectory = ([576] * 2 + [29] * 30) if method == "fastv" else (
                        [576] * 2 + [80] * 4 + [31] * 9 + [16] * 17
                    )
                    self.assertEqual(model.model.layer_visual_tokens, expected_trajectory)
                    self.assertAlmostEqual(model.model.visual_token_num, sum(expected_trajectory) / 32)
                    self.assertEqual(output.last_hidden_state.shape[1], prefix + expected_trajectory[-1] + 3)
                    cache = output.past_key_values
                    prefill_lengths = [value[0].shape[-2] for value in cache]
                    self.assertEqual(prefill_lengths, [prefix + count + 3 for count in expected_trajectory])
                    for step in range(1, 4):
                        output = model.model(
                            input_ids=torch.tensor([[12]]), past_key_values=cache,
                            attention_mask=torch.ones(1, original_length + step),
                            output_attentions=True, use_cache=True,
                        )
                        self.assertTrue(torch.isfinite(output.last_hidden_state).all())
                        self.assertEqual(output.last_hidden_state.shape, (1, 1, 64))
                        cache = output.past_key_values
                        self.assertEqual([value[0].shape[-2] for value in cache], [length + step for length in prefill_lengths])

    @torch.no_grad() if HAS_RUNTIME else (lambda fn: fn)
    def test_generate_integration_resets_between_different_prefixes(self):
        for method in ("fastv", "sparsevlm"):
            model = self.model(method)
            for prefix in (4, 15):
                with self.subTest(method=method, prefix=prefix):
                    ids = torch.tensor([[7] * prefix + [-200, 9, 10, 11]])
                    output, average = model.generate(
                        inputs=ids, images=torch.zeros(1, 3, 8, 8),
                        attention_mask=torch.ones_like(ids), texts="Describe the image.",
                        max_new_tokens=4, do_sample=False, use_cache=True,
                    )
                    self.assertEqual(output.shape[-1], 5)  # generated BOS plus four new tokens
                    self.assertEqual(model.model.system_prompt_length, prefix)
                    self.assertLess(average, 64)
                    self.assertGreater(average, 60)

    @torch.no_grad() if HAS_RUNTIME else (lambda fn: fn)
    def test_sdpa_13b_and_next_prefill_decode_with_per_layer_masks(self):
        for method in ("fastv", "sparsevlm"):
            for aspect, layers, crops in (("pad", 40, 1), ("anyres", 32, 5)):
                with self.subTest(method=method, aspect=aspect, layers=layers):
                    model = self.model(method, 64 * crops, aspect, layers, attention="sdpa")
                    self.assertTrue(model.model._use_sdpa)
                    packed = self.pack(model, prefix=8, crops=crops)
                    original_length = packed[4].shape[1]
                    output = model.model(
                        inputs_embeds=packed[4], attention_mask=packed[2], use_cache=True,
                    )
                    self.assertEqual(model.model.visual_token_length, 576 * crops)
                    self.assertEqual(len(model.model.layer_visual_tokens), layers)
                    self.assertLess(model.model.visual_token_num, 64 * crops)
                    self.assertGreater(model.model.visual_token_num, 60 * crops)
                    self.assertTrue(torch.isfinite(output.last_hidden_state).all())
                    cache = output.past_key_values
                    initial_lengths = [value[0].shape[-2] for value in cache]
                    for step in (1, 2):
                        output = model.model(
                            input_ids=torch.tensor([[12]]), past_key_values=cache,
                            attention_mask=torch.ones(1, original_length + step), use_cache=True,
                        )
                        cache = output.past_key_values
                        self.assertTrue(torch.isfinite(output.last_hidden_state).all())
                        self.assertEqual([value[0].shape[-2] for value in cache], [length + step for length in initial_lengths])

    @torch.no_grad() if HAS_RUNTIME else (lambda fn: fn)
    def test_explicit_padding_mask_survives_pruning_and_decode(self):
        for method in ("fastv", "sparsevlm"):
            with self.subTest(method=method):
                model = self.model(method)
                packed = self.pack(model, prefix=7)
                mask = packed[2].clone()
                mask[0, 0] = 0
                output = model.model(inputs_embeds=packed[4], attention_mask=mask, use_cache=True)
                original_length = packed[4].shape[1]
                self.assertTrue(all(not layer_mask[0, 0] for layer_mask in model.model._layer_padding_masks))
                output = model.model(
                    input_ids=torch.tensor([[12]]), past_key_values=output.past_key_values,
                    attention_mask=torch.cat((mask, torch.ones(1, 1)), dim=-1), use_cache=True,
                )
                self.assertTrue(torch.isfinite(output.last_hidden_state).all())
                self.assertTrue(all(not layer_mask[0, 0] for layer_mask in model.model._layer_padding_masks))

    @torch.no_grad() if HAS_RUNTIME else (lambda fn: fn)
    def test_fastv_rotary_cache_expands_for_original_positions(self):
        # First case crosses the initial rotary cache during prefill; second
        # crosses it on the first cached decode, with much shorter later layers.
        for initial_cache in (512, 586):
            with self.subTest(initial_cache=initial_cache):
                model = self.model("fastv", max_positions=initial_cache)
                packed = self.pack(model, prefix=7)
                output = model.model(inputs_embeds=packed[4], attention_mask=packed[2], use_cache=True)
                self.assertTrue(torch.isfinite(output.last_hidden_state).all())
                for step in range(1, 4):
                    output = model.model(
                        input_ids=torch.tensor([[12]]), past_key_values=output.past_key_values,
                        attention_mask=torch.ones(1, 586 + step), use_cache=True,
                    )
                    self.assertTrue(torch.isfinite(output.last_hidden_state).all())
                    self.assertGreaterEqual(model.model.layers[-1].self_attn.rotary_emb.max_seq_len_cached, 586 + step)


if __name__ == "__main__":
    unittest.main()
