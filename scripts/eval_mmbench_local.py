"""
本地 MMBench 评分脚本（无需 VLMEvalKit 的 build_dataset）

用法：

python scripts/eval_mmbench_local.py \
    --xlsx  /mnt/eason/LLaVA-STAR-Pro2/playground/data/eval/mmbench_cn/answers_upload/mmbench_dev_cn_20231003/llava-v1.5-7b/vscan/s1_144_s2_112.xlsx \
    --tsv   /mnt/eason_ckp/LLaVA-Eval/mmbench/mmbench_dev_20230712.tsv
python scripts/eval_mmbench_local.py \
      --xlsx  /mnt/eason/LLaVA-STAR-Pro2/playground/data/eval/mmbench_cn/answers_upload/mmbench_dev_cn_20231003/llava-v1.6-vicuna-13b/scope/vtn_640.xlsx \
      --tsv   /mnt/eason_ckp/LLaVA-Eval/mmbench_cn/mmbench_dev_cn_20231003.tsv

同时输出：
  - Vanilla accuracy   （与 MMBench 官方提交服务器口径一致）
  - Circular accuracy  （VLMEvalKit 口径，更严格）
  - 按 l2-category 细分（如 TSV 含该列）
"""

import argparse
import string
import pandas as pd


# ── 答案提取（完全对标 VLMEvalKit/vlmeval/utils/matching_util.py） ────────────

_REJECT_PHRASES = [
    "Sorry, I can't help with images of people yet.",
    "I can't process this file.",
    "I'm sorry, but without the image provided",
    'Cannot determine the answer',
]


def can_infer_option(pred: str, choices: dict) -> str | None:
    """对标 VLMEvalKit can_infer_option()。

    choices: {字母: 选项文字}，例如 {'A': 'cat', 'B': 'dog'}
    返回选项字母（或 'Z'），无法判断返回 None。
    """
    if 'Failed to obtain answer via API' in pred:
        return None

    for phrase in _REJECT_PHRASES:
        if phrase in pred:
            return 'Z'

    answer_mod = pred
    for ch in '.()[],:;!*#{}':
        answer_mod = answer_mod.replace(ch, ' ')

    splits = [x.strip() for x in answer_mod.split()]

    def count_in_splits(keys):
        return sum(1 for c in keys if c in splits)

    count = count_in_splits(choices)

    if count == 1:
        for ch in choices:
            if ch in splits:
                return ch
    elif count == 0 and count_in_splits({'Z', ''}) == 1:
        return 'Z'

    return None


def can_infer_text(pred: str, choices: dict) -> str | None:
    """对标 VLMEvalKit can_infer_text()。

    检查选项的完整文字是否出现在预测文本中。
    """
    pred_lower = pred.lower()
    cands = [k for k, v in choices.items() if str(v).lower() in pred_lower]
    return cands[0] if len(cands) == 1 else None


def can_infer(pred: str, choices: dict) -> str | None:
    """对标 VLMEvalKit can_infer()。

    第一阶段：字母精确匹配（can_infer_option）
    第二阶段：选项文本匹配（can_infer_text）
    两阶段均失败返回 None。
    """
    pred = str(pred)
    result = can_infer_option(pred, choices)
    return result if result else can_infer_text(pred, choices)


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def evaluate(xlsx_path: str, tsv_path: str):
    xlsx = pd.read_excel(xlsx_path, engine='openpyxl')
    tsv  = pd.read_csv(tsv_path, sep='\t')

    # 统一 index 类型
    xlsx['index'] = xlsx['index'].astype(int)
    tsv['index']  = tsv['index'].astype(int)

    # 确定 choices（选项字母列，如 A/B/C/D）
    choice_cols = [c for c in string.ascii_uppercase if c in xlsx.columns]

    # answer_map: index → 正确字母（来自 TSV）
    answer_map = dict(zip(tsv['index'], tsv['answer'].str.strip().str.upper()))

    # 过滤 xlsx 中在 answer_map 里的行
    xlsx = xlsx[xlsx['index'].isin(answer_map)].copy()
    xlsx['GT'] = xlsx['index'].map(answer_map)

    # 逐行提取预测选项（两阶段匹配，与官方完全一致）
    def get_pred(row):
        choices = {c: row[c] for c in choice_cols if not pd.isna(row.get(c, float('nan')))}
        return can_infer(str(row['prediction']), choices)

    xlsx['pred_raw'] = xlsx.apply(get_pred, axis=1)
    failed = int(xlsx['pred_raw'].isna().sum())

    # 对标官方 exact_matching 模式：提取失败视作 'Z'（算错）
    xlsx['pred'] = xlsx['pred_raw'].fillna('Z')
    xlsx['hit']  = (xlsx['pred'] == xlsx['GT'])

    # ── g_index（original question index）────────────────────────────────────
    xlsx['g_index'] = xlsx['index'] % 1_000_000

    # ── Vanilla accuracy（只看 g_index == index 的原始题，对标提交服务器） ──
    orig = xlsx[xlsx['index'] == xlsx['g_index']]
    vanilla_overall = orig['hit'].mean()

    print("=" * 55)
    print(f"  Vanilla  Overall : {vanilla_overall:.4f}  ({orig['hit'].sum()}/{len(orig)})")

    if 'l2-category' in tsv.columns:
        cat_map = dict(zip(tsv['index'], tsv['l2-category']))
        orig = orig.copy()
        orig['l2'] = orig['index'].map(cat_map)
        print()
        print("  Vanilla  by l2-category:")
        for cat, grp in sorted(orig.groupby('l2')):
            print(f"    {cat:<35s} {grp['hit'].mean():.4f}  ({grp['hit'].sum()}/{len(grp)})")

    # ── Circular accuracy（VLMEvalKit 口径：一组全对才算对） ─────────────────
    groups = xlsx.groupby('g_index')
    circ_hits = [int(g['hit'].all()) for _, g in groups]
    circular_overall = sum(circ_hits) / len(circ_hits) if circ_hits else 0.0

    print()
    print(f"  Circular Overall : {circular_overall:.4f}  ({sum(circ_hits)}/{len(circ_hits)})")
    print("=" * 55)

    if failed:
        print(f"  [warn] {failed} rows 无法提取选项字母（已按 exact_matching 算错）")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlsx', required=True, help='convert_mmbench_for_submission.py 生成的 xlsx')
    parser.add_argument('--tsv',  required=True, help='推理用的原始 TSV（同 --question-file）')
    args = parser.parse_args()
    evaluate(args.xlsx, args.tsv)
