import sys
import os

# 自动切换到 fa_xf 环境的 Python
_TARGET_PYTHON = '/mnt/eason/fa_xf/bin/python'
if sys.executable != _TARGET_PYTHON:
    os.execv(_TARGET_PYTHON, [_TARGET_PYTHON] + sys.argv)

import argparse
import string
import warnings

sys.path.insert(0, '/mnt/eason/VLMEvalKit-TOPS')
from vlmeval.dataset.utils.multiple_choice import mcq_circular_eval, report_acc
from vlmeval.smp import load_env
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--xlsx', type=str, default="/mnt/eason/LLaVA-STAR-Pro/playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712/llava-v1.5-7b/fastv/vtn_128.xlsx")
parser.add_argument('--judge', type=str, default='gpt-4o',
                    choices=['chatgpt-0125', 'gpt-4-0125', 'gpt-4o', 'exact_matching'],
                    help='评测用的 judge 模型 (默认: gpt-4o)')
parser.add_argument('--api-key', type=str, default="sk-xZHiJMncDzoyQhhcS9ho2xju1NKM2xiIMV1oJapMrXhgmPUS",
                    help='OpenAI API Key，也可通过环境变量 OPENAI_API_KEY 或 .env 文件设置')
parser.add_argument('--api-base', type=str, default="https://www.dmxapi.cn/v1/chat/completions",
                    help='OpenAI API Base URL（使用代理时设置）')
parser.add_argument('--dataset', type=str, default='mmbench_dev_20230712',
                    help='数据集名称，中文版传 MMBench_DEV_CN (默认: MMBench_DEV_EN)')
args = parser.parse_args()

# ── 设置 API Key ──────────────────────────────────────────────────────────────
if args.api_key:
    os.environ['OPENAI_API_KEY'] = args.api_key
if args.api_base:
    os.environ['OPENAI_API_BASE'] = args.api_base

# 尝试从 .env 文件加载（加载失败不影响后续逻辑）
load_env()

# ── 构建 judge 模型 ───────────────────────────────────────────────────────────
model = None
if args.judge != 'exact_matching':
    openai_key = os.environ.get('OPENAI_API_KEY', '')
    if isinstance(openai_key, str) and openai_key.startswith('sk-'):
        from vlmeval.dataset.utils.judge_util import build_judge
        print(f'[INFO] 使用 GPT judge: {args.judge}')
        try:
            model = build_judge(model=args.judge, nproc=4, verbose=False, retry=3)
            if not model.working():
                warnings.warn('OpenAI API 不可用，回退到 exact_matching 模式')
                model = None
        except Exception as e:
            warnings.warn(f'构建 judge 失败: {e}，回退到 exact_matching 模式')
            model = None
    else:
        warnings.warn('未检测到有效的 OPENAI_API_KEY，回退到 exact_matching 模式')
        warnings.warn('请通过 --api-key 参数或设置环境变量 OPENAI_API_KEY 提供 API Key')

if model is None:
    print('[INFO] 使用 exact_matching 模式（无 GPT）')

# ── 加载 xlsx ─────────────────────────────────────────────────────────────────
xlsx = args.xlsx
print(f'[INFO] 加载文件: {xlsx}')
data = pd.read_excel(xlsx, engine='openpyxl')
data['index'] = data['index'].astype(int)
data['prediction'] = [str(x) for x in data['prediction']]

# 列名归一化：非单个大写字母的列名转小写
for k in list(data.columns):
    if k not in list(string.ascii_uppercase):
        data = data.rename(columns={k: k.lower()})

meta = data[['index', 'answer']].copy()

# pkl 缓存放在 xlsx 同目录，按 judge 模式命名
judge_str = args.judge.replace('-', '_')
tmp_pkl = xlsx.replace('.xlsx', f'_{judge_str}_result.pkl')

# ── 评测 ──────────────────────────────────────────────────────────────────────
result = mcq_circular_eval(model, data, meta, nproc=4,
                           result_file=tmp_pkl,
                           dataset_name=args.dataset)
print(report_acc(result).to_string())
