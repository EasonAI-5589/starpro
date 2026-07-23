#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, use_text_tower=False, **kwargs):
    # Extract MustDrop configuration from kwargs
    use_mustdrop = kwargs.pop('use_mustdrop', False)
    mustdrop_config = kwargs.pop('mustdrop_config', {})

    # Extract VScan configuration from kwargs
    use_vscan = kwargs.pop('use_vscan', False)
    vscan_config = kwargs.pop('vscan_config', {})

    # Extract DUET-VLM configuration from kwargs (Stage-1 VisionZip + Stage-2 PyramidDrop)
    use_duet = kwargs.pop('use_duet', False)
    duet_config = kwargs.pop('duet_config', {})

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}
    elif device_map == "auto" and torch.cuda.device_count() == 1:
        # 单卡可见时（各 eval 脚本按 chunk 用 CUDA_VISIBLE_DEVICES 分卡）不能用 "auto"：
        # accelerate 会按加载瞬间的空闲显存决定是否 offload，邻居进程占用一高就把权重留在
        # meta 上，取值时报 "Cannot copy out of meta tensor"。模型远小于单卡显存，直接钉死。
        kwargs['device_map'] = {"": 0}
        device_map = {"": 0}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'qwen' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                # 🔥 MustDrop: Load MustDrop-enabled model when use_mustdrop=True
                # Reference: MustDrop official builder.py:121-127
                if use_mustdrop:
                    print("Loading MustDrop LLaVA model (Dual Attention Filter)...")
                    # 🔥 CRITICAL: Set use_mustdrop in config BEFORE from_pretrained()
                    # Vision Tower is initialized during from_pretrained(), so config must be ready
                    from transformers import AutoConfig
                    mustdrop_cfg = AutoConfig.from_pretrained(model_path)
                    mustdrop_cfg.use_mustdrop = True
                    # 🔥 Fix transformers 4.37+ compatibility: add missing config attributes
                    # Old LLaVA checkpoints may lack these fields required by newer transformers
                    if not hasattr(mustdrop_cfg, 'attention_dropout'):
                        mustdrop_cfg.attention_dropout = 0.0
                    if not hasattr(mustdrop_cfg, 'rope_theta'):
                        mustdrop_cfg.rope_theta = 10000.0
                    if not hasattr(mustdrop_cfg, 'rope_scaling'):
                        mustdrop_cfg.rope_scaling = None
                    if not hasattr(mustdrop_cfg, 'attention_bias'):
                        mustdrop_cfg.attention_bias = False
                    # 🔥 CRITICAL: MustDrop requires SDPA attention implementation for performance
                    # Without this, it falls back to eager mode which is ~10x slower
                    mustdrop_cfg._attn_implementation = 'sdpa'
                    model = MustDropLlavaLlamaForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        config=mustdrop_cfg,
                        **kwargs
                    )
                    # Store MustDrop config for later use in generate()
                    model.mustdrop_config = mustdrop_config
                # 🔬 VScan: Load VScan-enabled model when use_vscan=True
                # Reference: https://github.com/Tencent/SelfEvolvingAgent/tree/main/VScan
                elif use_vscan:
                    print("Loading VScan LLaVA model (Training-Free Visual Token Reduction)...")
                    from transformers import AutoConfig
                    vscan_cfg = AutoConfig.from_pretrained(model_path)
                    # Fix transformers 4.37+ compatibility
                    if not hasattr(vscan_cfg, 'attention_dropout'):
                        vscan_cfg.attention_dropout = 0.0
                    if not hasattr(vscan_cfg, 'rope_theta'):
                        vscan_cfg.rope_theta = 10000.0
                    if not hasattr(vscan_cfg, 'rope_scaling'):
                        vscan_cfg.rope_scaling = None
                    if not hasattr(vscan_cfg, 'attention_bias'):
                        vscan_cfg.attention_bias = False
                    vscan_cfg._attn_implementation = 'sdpa'

                    # Set VScan-specific config parameters
                    vscan_cfg.use_vscan = True
                    if 'stage1_tokens' not in vscan_config:
                        vscan_config['stage1_tokens'] = 96
                    if 'stage2_tokens' not in vscan_config:
                        vscan_config['stage2_tokens'] = 32
                    if 'prune_layer' not in vscan_config:
                        num_layers = vscan_cfg.num_hidden_layers
                        vscan_config['prune_layer'] = 16 if num_layers == 32 else 20

                    model = LlavaLlamaForCausalLM_VScan.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        config=vscan_cfg,
                        vscan_config=vscan_config,
                        **kwargs
                    )
                    # Store VScan config for later use
                    model.vscan_config = vscan_config
                    print(f"  Stage 1 tokens: {vscan_config['stage1_tokens']}")
                    print(f"  Stage 2 tokens: {vscan_config['stage2_tokens']}")
                    print(f"  Prune layer: {vscan_config['prune_layer']}")
                # 🔬 DUET-VLM (AMD-AGI, CVPR2026): Stage-2 class = PyramidDrop; Stage-1
                # VisionZip is applied post-load below. Reference: DUET-VLM/llava/model/builder.py
                elif use_duet:
                    print("Loading DUET-VLM LLaVA model (VisionZip + PyramidDrop)...")
                    from transformers import AutoConfig
                    duet_cfg = AutoConfig.from_pretrained(model_path)
                    # Fix transformers 4.37+ compatibility (old LLaVA ckpts lack these)
                    if not hasattr(duet_cfg, 'attention_dropout'):
                        duet_cfg.attention_dropout = 0.0
                    if not hasattr(duet_cfg, 'rope_theta'):
                        duet_cfg.rope_theta = 10000.0
                    if not hasattr(duet_cfg, 'rope_scaling'):
                        duet_cfg.rope_scaling = None
                    if not hasattr(duet_cfg, 'attention_bias'):
                        duet_cfg.attention_bias = False
                    duet_cfg._attn_implementation = 'sdpa'
                    # DUET config knob (checkpoints don't carry it, so inject here)
                    duet_cfg.use_salient_tokens = duet_config.get('use_salient_tokens', False)
                    model = LlavaLlamaForCausalLM_Duet.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        config=duet_cfg,
                        **kwargs
                    )
                    # Store DUET config; VisionZip + schedule applied after vision tower load
                    model.duet_config = duet_config
                else:
                    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if use_text_tower:
            vision_tower.load_text_tower(device_map=device_map)
        if device_map != 'auto':
            # device_map 可能是 {"": 0} 这样的 dict，.to() 只接受具体设备
            target_device = device_map[""] if isinstance(device_map, dict) else device_map
            vision_tower.to(device=target_device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    # 🔬 VScan: Additional runtime configuration (if model was loaded with VScan)
    # Note: Main VScan config is now set during model loading above
    if use_vscan and hasattr(model, 'vscan_config'):
        # Set up Stage 2 parameters on model.model for runtime
        model.model.layer_list = [vscan_config['prune_layer']]
        model.model.image_token_list = [vscan_config['stage1_tokens'], vscan_config['stage2_tokens']]
        model.model.visual_token_length = vscan_config['stage1_tokens']
        # Use AVERAGE visual tokens for display: (stage1 + stage2) / 2
        model.model.visual_token_num = (vscan_config['stage1_tokens'] + vscan_config['stage2_tokens']) // 2

    # 🔬 DUET: Stage-1 VisionZip must be applied AFTER the CLIP tower is loaded
    # (visionzip() dereferences model.model.vision_tower.vision_tower), then set the
    # Stage-2 PyramidDrop layer schedule. Reference: DUET-VLM/llava/eval/model_vqa_loader.py
    if use_duet and hasattr(model, 'duet_config'):
        from visionzip import visionzip
        dominant = duet_config.get('dominant', 300)
        contextual = duet_config.get('contextual', 7)
        cluster_width = duet_config.get('cluster_width', 4)
        model = visionzip(model, dominant=dominant, contextual=contextual, cluster_width=cluster_width)
        model.model.layer_list = list(duet_config.get('layer_list', [16, 24]))
        ratios = list(duet_config.get('image_token_ratio_list', [0.5, 0.0]))
        ratios.insert(0, 1.0)
        model.model.image_token_ratio_list = ratios
        # Average-over-layers visual-token budget label (display only)
        model.model.visual_token_num = duet_config.get('visual_token_num', None)
        print(f"  VisionZip: dominant={dominant} contextual={contextual} cluster_width={cluster_width}")
        print(f"  PyramidDrop: layer_list={model.model.layer_list} image_token_ratio_list={model.model.image_token_ratio_list}")

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
