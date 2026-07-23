"""
Debug script: find exact location of float/half dtype mismatch in cdp3.
Patches F.linear to print full traceback on dtype mismatch.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import traceback as tb

# ── patch F.linear to catch dtype mismatch ──────────────────────────────────
_orig_linear = F.linear
_mismatch_count = [0]

def _debug_linear(input, weight, bias=None):
    if input.dtype != weight.dtype:
        _mismatch_count[0] += 1
        if _mismatch_count[0] <= 3:          # only print first 3 to avoid spam
            print(f"\n{'='*70}")
            print(f"[DEBUG] F.linear dtype mismatch #{_mismatch_count[0]}")
            print(f"  input  : dtype={input.dtype}  shape={tuple(input.shape)}")
            print(f"  weight : dtype={weight.dtype}  shape={tuple(weight.shape)}")
            tb.print_stack(limit=20)
        raise RuntimeError(
            f"expected mat1 and mat2 to have the same dtype, "
            f"but got: {input.dtype} != {weight.dtype}"
        )
    return _orig_linear(input, weight, bias)

F.linear = _debug_linear
torch.nn.functional.linear = _debug_linear

# ── load model ───────────────────────────────────────────────────────────────
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from PIL import Image

disable_torch_init()
model_path = '/mnt/eason_ckp/models/llava-v1.5-7b'
model_name  = get_model_name_from_path(model_path)

tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path, None, model_name,
    pruning_method='cdp3',
    visual_token_num=64,
    use_text_tower=True,
    device_map='cuda:0',
)
model.eval()
print("Model loaded OK")

# ── run one sample ────────────────────────────────────────────────────────────
import glob
imgs = sorted(glob.glob('/mnt/eason_ckp/LLaVA-Eval/textvqa/train_images/*.jpg'))
img_path = imgs[0]
question  = 'What is the text in the image?'

image        = Image.open(img_path).convert('RGB')
image_tensor = process_images([image], image_processor, model.config)[0]
image_tensor = image_tensor.unsqueeze(0).to(dtype=torch.float16, device=model.device)

qs    = DEFAULT_IMAGE_TOKEN + '\n' + question
conv  = conv_templates['vicuna_v1'].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt    = conv.get_prompt()
input_ids = tokenizer_image_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
).unsqueeze(0).to(model.device)

with torch.inference_mode():
    out = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=False,
        max_new_tokens=32,
        use_cache=True,
        texts=question,
    )

if isinstance(out, tuple): out = out[0]
answer = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
print(f"Success!  Answer: {answer}")
