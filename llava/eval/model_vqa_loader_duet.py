# ------------------------------------------------------------------------
# DUET-VLM (AMD-AGI, CVPR 2026) eval loader for LLaVA-STAR-Pro2.
# Isolated from the shared model_vqa_loader.py so DUET's distinct generate
# signature (single return + idxs=salient) and salient-token computation do
# not touch the 20+ other baselines.
#
# Stage-1 VisionZip + Stage-2 PyramidDrop schedule are applied inside
# llava/model/builder.py (use_duet path). This loader only:
#   - loads the model with use_duet=True,
#   - optionally computes salient text-token indices (T2V) per sample,
#   - calls model.generate(..., idxs=salient) and decodes.
# ------------------------------------------------------------------------
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, compute_salient_tokens=False):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.compute_salient_tokens = compute_salient_tokens
        if self.compute_salient_tokens:
            # Lazy import so non-salient runs (and non-duet envs) don't need spacy.
            from llava.salient_token_finder import salient_tokens_finder
            from llava.aligner import greedy_align_and_filter
            from llava.duet_lcs import longest_common_subarray
            self._salient_tokens_finder = salient_tokens_finder
            self._greedy_align_and_filter = greedy_align_and_filter
            self._longest_common_subarray = longest_common_subarray

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        original_prompt = qs

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        salient_token_indices = None
        if self.compute_salient_tokens:
            # Ported from DUET-VLM/llava/eval/model_vqa_loader.py: build salient
            # text-token indices for T2V-guided PyramidDrop.
            salient_tokens_filtered = [t.lower() for t in self._salient_tokens_finder(original_prompt)]
            res = self._longest_common_subarray(self.tokenizer(original_prompt).input_ids, input_ids.detach().cpu().tolist())
            if res is not None:
                qqs_start = res['start_in_qqs_0based']
                prompt_start = res['start_in_prompt_0based']
                if qqs_start > 1:
                    rel_seq = [ele for ele in input_ids[prompt_start - (qqs_start - 1):prompt_start]] + res['match']
                    start = prompt_start - (qqs_start - 1)
                else:
                    rel_seq = res['match']
                    start = prompt_start
            else:
                raise NotImplementedError("No match found for salient token alignment")
            rel_words = [self.tokenizer.decode([w], skip_special_tokens=True) for w in rel_seq]
            assert len(rel_seq) == len(rel_words), "Length mismatch"
            salient_token_original = self._greedy_align_and_filter(rel_words, salient_tokens_filtered)
            salient_token_indices = [start + id for id, word in enumerate(rel_words) if word in salient_token_original]
            # Always keep the final prompt token (the ':' before the answer).
            salient_token_indices.append(len(input_ids) - 1)

        return input_ids, image_tensor, image.size, salient_token_indices

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, salient_tokens = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes, salient_tokens


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, compute_salient_tokens, batch_size=1, num_workers=2):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, compute_salient_tokens)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def _parse_list_arg(s, default):
    if s is None:
        return list(default)
    return list(eval(s))


def eval_model(args):
    if os.environ.get('ENABLE_DEBUG', '0') != '1':
        import warnings
        warnings.filterwarnings('ignore')

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # ===== DUET config (VisionZip stage-1 + PyramidDrop stage-2) =====
    duet_config = {
        "dominant": args.dominant,
        "contextual": args.contextual,
        "cluster_width": args.cluster_width,
        "layer_list": _parse_list_arg(args.layer_list, [16, 24]),
        "image_token_ratio_list": _parse_list_arg(args.image_token_ratio_list, [0.5, 0.0]),
        "use_salient_tokens": bool(args.compute_salient_tokens),
        "visual_token_num": args.visual_token_num,
    }

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        pruning_method="duet",
        visual_token_num=args.visual_token_num,
        use_duet=True, duet_config=duet_config,
    )

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    # salient-token flag: CLI overrides, else model config
    compute_salient_tokens = bool(args.compute_salient_tokens)
    if not compute_salient_tokens:
        compute_salient_tokens = getattr(model.config, 'use_salient_tokens', False)
    print(f"[DUET] Using salient tokens (T2V): {compute_salient_tokens}")

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, compute_salient_tokens)

    show_progress = os.environ.get('ENABLE_DEBUG', '0') == '1'
    data_bar = tqdm(zip(data_loader, questions), total=len(questions), disable=not show_progress)
    for (input_ids, image_tensors, image_sizes, salient_tokens), line in data_bar:
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensors = image_tensors.to(dtype=torch.float16, device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                idxs=salient_tokens[0],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        vtn = getattr(model.model, 'visual_token_num', None)
        data_bar.set_postfix(vtn=vtn)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--pruning_method", type=str, default="duet")
    parser.add_argument("--visual_token_num", type=int, default=192)

    # DUET / VisionZip stage-1
    parser.add_argument("--dominant", type=int, default=300)
    parser.add_argument("--contextual", type=int, default=7)
    parser.add_argument("--cluster_width", type=int, default=4)
    # DUET / PyramidDrop stage-2
    parser.add_argument("--layer_list", type=str, default="[16,24]")
    parser.add_argument("--image_token_ratio_list", type=str, default="[0.5,0.0]")
    parser.add_argument("--compute_salient_tokens", action='store_true', default=False)

    args = parser.parse_args()

    eval_model(args)
