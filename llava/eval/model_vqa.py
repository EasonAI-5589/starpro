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

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    use_fastv = True if args.pruning_method == "fastv" else False
    fastv_config = {"K": 2, "T": args.visual_token_num}
    use_sparsevlm = True if args.pruning_method == "sparsevlm" else False
    sparsevlm_config = {"T": args.visual_token_num}
    use_d2p = True if args.pruning_method == "d2p" else False
    d2p_config = {"T": args.visual_token_num}
    use_svdvlm = True if args.pruning_method == "svdvlm" else False
    svdvlm_config = {"T": args.visual_token_num}
    use_prefixvlm = True if args.pruning_method == "prefixvlm" else False
    prefixvlm_config = {"T": args.visual_token_num}
    use_holov2 = True if args.pruning_method == "HoloV_2" else False
    holov2_config = {"T": args.visual_token_num}
    use_idea = True if args.pruning_method == "Idea" else False
    idea_config = {"T": args.visual_token_num}
    use_prefixvlm_2 = True if args.pruning_method == "prefixvlm_2" else False
    prefixvlm_2_config = {"T": args.visual_token_num}
    use_pdrop = True if args.pruning_method == "pdrop" else False
    pdrop_config = {"T": args.visual_token_num}

    # VScan Configuration
    use_vscan = True if args.pruning_method == "vscan" else False
    vscan_config = {
        "stage1_tokens": args.visual_token_num,  # Stage 1 output
        "stage2_tokens": args.vscan_stage2_tokens,  # Stage 2 output
        "prune_layer": args.vscan_prune_layer,  # Layer for Stage 2 pruning
    }

    # STAR Configuration
    use_star = True if args.pruning_method in ["star", "star_v2", "star_pro"] else False
    star_config = {
        "T": args.visual_token_num,
        "num_latent": 20,
        "latent_pool_size": (5, 4),
        "mode": "star_pro" if args.pruning_method == "star_pro" else ("star_v2" if args.pruning_method == "star_v2" else "star"),
        "debug": True,
    }

    # 🔥 MustDrop Configuration (from model_vqa_loader.py)
    MUSTDROP_THRESHOLDS = {
        64:  {"global_thr": 0.011,  "individual_thr": 0.01},
        128: {"global_thr": 0.0012, "individual_thr": 0.001},
        192: {"global_thr": 0.001,  "individual_thr": 0.001},
    }
    use_mustdrop = True if args.pruning_method == "mustdrop" else False
    thr_config = MUSTDROP_THRESHOLDS.get(args.visual_token_num, MUSTDROP_THRESHOLDS[128])
    mustdrop_config = {
        "pruning_layers": [2, 6, 10, 14],
        "global_thr": thr_config["global_thr"],
        "individual_thr": thr_config["individual_thr"],
        "keep_rate": 0.08,
        "merge_threshold": 0.8,
        "merge_window_size": (3, 3),
        "T": args.visual_token_num,
    }

    use_text_tower = True if args.pruning_method == "trim" or "cdp3" in args.pruning_method or "thcp" in args.pruning_method or args.pruning_method == "star_pro" or args.pruning_method == "prefixvlm" else False
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        pruning_method=args.pruning_method,
        visual_token_num=args.visual_token_num,
        use_fastv=use_fastv, fastv_config=fastv_config,
        use_sparsevlm=use_sparsevlm, sparsevlm_config=sparsevlm_config,
        use_d2p=use_d2p, d2p_config=d2p_config,
        use_svdvlm=use_svdvlm, svdvlm_config=svdvlm_config,
        use_prefixvlm=use_prefixvlm, prefixvlm_config=prefixvlm_config,
        use_holov2=use_holov2, holov2_config=holov2_config,
        use_idea=use_idea, idea_config=idea_config,
        use_prefixvlm_2=use_prefixvlm_2, prefixvlm_2_config=prefixvlm_2_config,
        use_pdrop=use_pdrop, pdrop_config=pdrop_config,
        use_star=use_star, star_config=star_config,
        use_mustdrop=use_mustdrop, mustdrop_config=mustdrop_config,
        use_vscan=use_vscan, vscan_config=vscan_config,
        use_text_tower=use_text_tower,
    )

    # Data
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_bar = tqdm(questions)
    for line in data_bar:
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        question = cur_prompt

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            # 🔥 MustDrop: Different generate() signature (from model_vqa_loader.py)
            if use_mustdrop:
                output_ids = model.generate(
                    mustdrop_config["global_thr"],      # First positional arg
                    mustdrop_config["individual_thr"],  # Second positional arg
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True)
                visual_token_num = getattr(model.model, 'visual_token_num', 576)
            else:
                output_ids, visual_token_num = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    texts=question,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)
                if hasattr(model.model, 'visual_token_num'):
                    visual_token_num = model.model.visual_token_num
            data_bar.set_postfix(vtn=f"{visual_token_num}")

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
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--pruning_method", type=str, default=None)
    parser.add_argument("--visual_token_num", type=int, default=576)
    parser.add_argument("--vscan_stage2_tokens", type=int, default=32,
                        help="VScan stage 2 output tokens")
    parser.add_argument("--vscan_prune_layer", type=int, default=16,
                        help="VScan stage 2 pruning layer")
    args = parser.parse_args()

    eval_model(args)
