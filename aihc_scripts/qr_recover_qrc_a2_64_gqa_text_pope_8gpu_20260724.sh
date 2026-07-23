#!/usr/bin/env bash
set -uo pipefail

ROOT=/mnt/eason/LLaVA-STAR-Pro2-qr_recover_20260723
TAG="${STARPRO_RUN_TAG:-llavaqrc_a2_64_gqa_text_pope_8gpu_20260724}"
LOG_ROOT="${STARPRO_LOG_ROOT:-$ROOT/results_aihc_qrc_a2_64_gqa_text_pope_20260724}"
OUT="$LOG_ROOT/$TAG"
SCORERS="${STARPRO_SCORERS:-qr_centered qr}"
BENCHES="${STARPRO_BENCHES:-gqa textvqa pope}"
mkdir -p "$OUT"

source /mnt/eason/miniconda3/etc/profile.d/conda.sh
conda activate llava

export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CKPT_DIR="${CKPT_DIR:-/mnt/eason_ckp/models}"
export DATA_DIR="${DATA_DIR:-/mnt/eason_ckp/LLaVA-Eval}"
export ENABLE_DEBUG=1
export STAR_KEEP_POSIDS=1
export STAR_CAUSAL_FIX=1
export STAGE1_MULT=2
export TEXT_AGG_MODE=average_all
export METHOD=star_pro
export TOKEN=64
export VISUAL_TOKEN_NUM="$TOKEN"
unset LAMBDA STAGE1_SVD_RANK CUSTOM_PRUNING_SCHEDULE

cd "$ROOT"

for scorer in $SCORERS; do
  export STAGE1_SCORER="$scorer"
  SUBOUT="$OUT/$scorer"
  mkdir -p "$SUBOUT"
  export RUN_TAG="${TAG}_${scorer}"
  export RESULT_DIR="$SUBOUT"
  {
    echo "CONFIG tag=$TAG scorer=$scorer METHOD=$METHOD TOKEN=$TOKEN VISUAL_TOKEN_NUM=$VISUAL_TOKEN_NUM STAGE1_MULT=$STAGE1_MULT TEXT_AGG_MODE=$TEXT_AGG_MODE STAR_KEEP_POSIDS=$STAR_KEEP_POSIDS STAR_CAUSAL_FIX=$STAR_CAUSAL_FIX CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    for bench in $BENCHES; do
      echo "===== BEGIN $TAG $scorer $bench ====="
      bash "scripts/v1_5/7b/${bench}.sh" "$METHOD" "$TOKEN" 2>&1 | tee "$SUBOUT/${bench}.fulllog"
      echo "===== END $TAG $scorer $bench ====="
    done
  } 2>&1 | tee "$SUBOUT/suite.log"
done

STARPRO_OUT="$OUT" STARPRO_SCORERS="$SCORERS" STARPRO_BENCHES="$BENCHES" python3 - <<'PY'
import os, re, sys
root=os.environ["STARPRO_OUT"]; scorers=os.environ["STARPRO_SCORERS"].split(); benches=os.environ["STARPRO_BENCHES"].split()
bad=("Traceback","RuntimeError","CUDA out of memory","FileNotFoundError","[ERROR]")
pats={"gqa":r"Accuracy:\s*([\d.]+)%","textvqa":r"Accuracy:\s*([\d.]+)%","pope":r"Average F1 score:\s*([\d.]+)"}
res={}; failed=False
print("=================== QRC PAIRED RESULT ===================")
for scorer in scorers:
    res[scorer]={}
    for bench in benches:
        p=os.path.join(root,scorer,f"{bench}.fulllog")
        if not os.path.exists(p): print(f"{scorer}/{bench}: MISSING_LOG"); failed=True; continue
        t=open(p,errors="replace").read()
        s1=t.count(f"scorer={scorer}"); cen=t.count("qr_centered: mean-centered features"); causal=t.count("[STAR-PRO CAUSAL FIX]"); aa=t.count("Average-all (")
        hit=[w for w in bad if w in t]
        need_cen = (scorer=="qr_centered")
        if hit: print(f"{scorer}/{bench}: ERROR={hit} s1={s1} cen={cen} causal={causal} avgall={aa}"); failed=True; continue
        if s1<=0 or causal<=0 or aa<=0 or (need_cen and cen<=0):
            print(f"{scorer}/{bench}: CONFIG_NOT_PROVEN s1={s1} cen={cen} causal={causal} avgall={aa}"); failed=True; continue
        m=re.findall(pats[bench],t)
        if not m: print(f"{scorer}/{bench}: MISSING_SCORE s1={s1}"); failed=True; continue
        v=float(m[-1]); v = v*100.0 if bench=="pope" else v
        res[scorer][bench]=v
        print(f"{scorer}/{bench}: {v:.4f}  (s1={s1} cen={cen} causal={causal} avgall={aa})")
print("--------------------- DELTA (qr_centered - qr) ---------------------")
for bench in benches:
    a=res.get("qr_centered",{}).get(bench); b=res.get("qr",{}).get(bench)
    if a is not None and b is not None: print(f"{bench}: qr_centered={a:.4f}  qr={b:.4f}  delta={a-b:+.4f}")
print("========================================================")
sys.exit(80 if failed else 0)
PY

echo "ALL_DONE llava_qrc_a2_64_gqa_text_pope_8gpu_20260724"
