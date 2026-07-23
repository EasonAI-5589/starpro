# ------------------------------------------------------------------------
# DUET-VLM integration helper: longest_common_subarray + to_list.
# Ported verbatim from DUET-VLM/llava/utils.py so we don't have to touch our
# own llava/utils.py. Used by the DUET eval loader to align salient words to
# input-id positions for the Text-to-Vision (T2V) PyramidDrop stage.
# ------------------------------------------------------------------------
from typing import Optional, Dict, Any, Sequence


def to_list(x):
    """Convert torch tensor / numpy array / list-like to Python list of ints."""
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(x, torch.Tensor):
        return x.tolist()
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None and isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def longest_common_subarray(qqs: Sequence[int],
                            prompt: Sequence[int]
                            ) -> Optional[Dict[str, Any]]:
    """
    Full-DP (O(n*m) time & space) longest contiguous common subarray.
    If multiple matches have the same max length, return the one with the
    earliest start index in `prompt` (the second argument).
    """
    if not isinstance(qqs, list):
        qqs = to_list(qqs)
    if not isinstance(prompt, list):
        prompt = to_list(prompt)

    if not qqs or not prompt:
        return None

    n, m = len(qqs), len(prompt)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    max_len = 0
    end_a = -1
    end_b = -1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if qqs[i - 1] == prompt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                cur_len = dp[i][j]
                cur_start_b = j - cur_len
                if cur_len > max_len:
                    max_len = cur_len
                    end_a = i - 1
                    end_b = j - 1
                elif cur_len == max_len and cur_len > 0:
                    prev_start_b = end_b - max_len + 1
                    if cur_start_b < prev_start_b:
                        end_a = i - 1
                        end_b = j - 1
            else:
                dp[i][j] = 0

    if max_len == 0:
        return None

    start_a = end_a - max_len + 1
    start_b = end_b - max_len + 1
    match = qqs[start_a:end_a + 1]

    return {
        "length": max_len,
        "start_in_qqs_0based": start_a,
        "end_in_qqs_0based": end_a,
        "start_in_prompt_0based": start_b,
        "end_in_prompt_0based": end_b,
        "start_in_qqs_1based": start_a + 1,
        "end_in_qqs_1based": end_a + 1,
        "start_in_prompt_1based": start_b + 1,
        "end_in_prompt_1based": end_b + 1,
        "match": match
    }
