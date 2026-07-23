#!/usr/bin/env bash
set -uo pipefail

ROOT=/mnt/eason/LLaVA-STAR-Pro2-qr_recover_20260723
TAG="${STARPRO_RUN_TAG:-llavaqr_schedule_shape_64_20260724}"
LOG_ROOT="${STARPRO_LOG_ROOT:-$ROOT/results_aihc_qr_schedule_64_20260724}"
OUT="$LOG_ROOT/$TAG"
BENCHES="${STARPRO_BENCHES:-gqa textvqa pope}"
mkdir -p "$OUT"

source /mnt/eason/miniconda3/etc/profile.d/conda.sh
conda activate llava

export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CKPT_DIR="${CKPT_DIR:-/mnt/eason_ckp/models}"
export DATA_DIR="${DATA_DIR:-/mnt/eason_ckp/LLaVA-Eval}"
export ENABLE_DEBUG=1 STAR_KEEP_POSIDS=1 STAR_CAUSAL_FIX=1
export STAGE1_SCORER=qr STAGE1_MULT=2 STAGE2_ANCHOR_M=0 TEXT_AGG_MODE=average_all
export METHOD=star_pro TOKEN=64 VISUAL_TOKEN_NUM=64
unset LAMBDA STAGE1_SVD_RANK

cd "$ROOT"

# name -> schedule JSON (all time-avg = 64.0, iso-FLOPs; Stage-1 K=128)
declare -A SCHED
SCHED[SA_early1]='[[2,60]]'
SCHED[SB_default2]='[[12,32],[24,16]]'
SCHED[SC_grad3]='[[8,64],[16,40],[24,24]]'
SCHED[SE_grad5]='[[4,96],[10,64],[16,48],[22,32],[28,24]]'

for arm in SA_early1 SB_default2 SC_grad3 SE_grad5; do
  export CUSTOM_PRUNING_SCHEDULE="${SCHED[$arm]}"
  SUBOUT="$OUT/$arm"; mkdir -p "$SUBOUT"
  export RUN_TAG="${TAG}_${arm}"; export RESULT_DIR="$SUBOUT"
  {
    echo "CONFIG tag=$TAG arm=$arm CUSTOM_PRUNING_SCHEDULE=${SCHED[$arm]} STAGE1_MULT=2 STAGE2_ANCHOR_M=0 TEXT_AGG_MODE=average_all"
    for bench in $BENCHES; do
      echo "===== BEGIN $TAG $arm $bench ====="
      bash "scripts/v1_5/7b/${bench}.sh" "$METHOD" "$TOKEN" 2>&1 | tee "$SUBOUT/${bench}.fulllog"
      echo "===== END $TAG $arm $bench ====="
    done
  } 2>&1 | tee "$SUBOUT/suite.log"
done

STARPRO_OUT="$OUT" STARPRO_BENCHES="$BENCHES" python3 - <<'PY'
import os, re
root=os.environ["STARPRO_OUT"]; benches=os.environ["STARPRO_BENCHES"].split()
arms=["SA_early1","SB_default2","SC_grad3","SE_grad5"]
pats={"gqa":r"Accuracy:\s*([\d.]+)%","textvqa":r"Accuracy:\s*([\d.]+)%","pope":r"Average F1 score:\s*([\d.]+)"}
print("======= SCHEDULE SHAPE (iso-avg=64, K=128, M=0, average_all) =======")
print("%-13s %8s %8s %8s"%("arm","GQA","TextVQA","POPE"))
res={}
for a in arms:
    row={}
    for b in benches:
        p=os.path.join(root,a,b+".fulllog")
        if os.path.exists(p):
            t=open(p,errors="replace").read(); m=re.findall(pats[b],t)
            if m: v=float(m[-1]); row[b]=v*100 if b=="pope" else v
    res[a]=row
    def f(b): return ("%.2f"%row[b]) if b in row else "--"
    print("%-13s %8s %8s %8s"%(a,f("gqa"),f("textvqa"),f("pope")))
print("--- SA=1 early hard cut | SB=default 2 cuts | SC=3 gradual | SE=5 gradual (abstiche/peeling) ---")
print("--- FLAT across shapes => schedule doesn't matter; gradual>default => peeling helps ---")
PY

echo "ALL_DONE llava_qr_schedule_shape_64_20260724"
