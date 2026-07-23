#!/usr/bin/env bash
set -uo pipefail

ROOT=/mnt/eason/LLaVA-STAR-Pro2-qr_recover_20260723
TAG="${STARPRO_RUN_TAG:-llavaqr_flatness_a_sweep_64_20260724}"
LOG_ROOT="${STARPRO_LOG_ROOT:-$ROOT/results_aihc_qr_flatness_64_20260724}"
OUT="$LOG_ROOT/$TAG"
# candidate-count sweep via STAGE1_MULT (K = 64 * MULT): 64,96,128,192,288
MULTS="${STARPRO_MULTS:-1 1.5 2 3 4.5}"
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
export STAGE1_SCORER=qr
export STAGE2_ANCHOR_M=0
export TEXT_AGG_MODE=average_all
export METHOD=star_pro
export TOKEN=64
export VISUAL_TOKEN_NUM="$TOKEN"
unset LAMBDA STAGE1_SVD_RANK CUSTOM_PRUNING_SCHEDULE

cd "$ROOT"

for mult in $MULTS; do
  export STAGE1_MULT="$mult"
  K=$(python3 -c "print(int(round($TOKEN*$mult)))")
  SUBOUT="$OUT/mult_${mult}_K${K}"
  mkdir -p "$SUBOUT"
  export RUN_TAG="${TAG}_m${mult}"
  export RESULT_DIR="$SUBOUT"
  {
    echo "CONFIG tag=$TAG STAGE1_MULT=$mult K=$K STAGE2_ANCHOR_M=0 METHOD=$METHOD TOKEN=$TOKEN TEXT_AGG_MODE=average_all"
    for bench in $BENCHES; do
      echo "===== BEGIN $TAG K=$K $bench ====="
      bash "scripts/v1_5/7b/${bench}.sh" "$METHOD" "$TOKEN" 2>&1 | tee "$SUBOUT/${bench}.fulllog"
      echo "===== END $TAG K=$K $bench ====="
    done
  } 2>&1 | tee "$SUBOUT/suite.log"
done

STARPRO_OUT="$OUT" STARPRO_MULTS="$MULTS" STARPRO_BENCHES="$BENCHES" STARPRO_TOKEN="$TOKEN" python3 - <<'PY'
import os, re, glob
root=os.environ["STARPRO_OUT"]; mults=os.environ["STARPRO_MULTS"].split(); benches=os.environ["STARPRO_BENCHES"].split(); T=int(os.environ["STARPRO_TOKEN"])
pats={"gqa":r"Accuracy:\s*([\d.]+)%","textvqa":r"Accuracy:\s*([\d.]+)%","pope":r"Average F1 score:\s*([\d.]+)"}
print("=========== FLATNESS: accuracy vs candidate-count K (T=64, M=0, average_all; compute varies) ===========")
hdr="%-6s %-5s %8s %8s %8s"%("MULT","K","GQA","TextVQA","POPE"); print(hdr)
res={}
for mult in mults:
    K=int(round(T*float(mult)))
    d=glob.glob(os.path.join(root,"mult_%s_K%d"%(mult,K)))
    d=d[0] if d else os.path.join(root,"mult_%s_K%d"%(mult,K))
    row={}
    for b in benches:
        p=os.path.join(d,b+".fulllog")
        if os.path.exists(p):
            t=open(p,errors="replace").read(); m=re.findall(pats[b],t)
            if m:
                v=float(m[-1]); v=v*100 if b=="pope" else v; row[b]=v
    res[(mult,K)]=row
    def f(b): return ("%.2f"%row[b]) if b in row else "--"
    print("%-6s %-5d %8s %8s %8s"%(mult,K,f("gqa"),f("textvqa"),f("pope")))
print("--- read: FLAT across K => 2T(128) arbitrary; RISING => K is a real tradeoff; PEAK => optimal K exists ---")
PY

echo "ALL_DONE llava_qr_flatness_a_sweep_64_20260724"
