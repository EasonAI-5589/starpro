#!/usr/bin/env bash
set -uo pipefail

ROOT=/mnt/eason/LLaVA-STAR-Pro2-qr_recover_20260723
TAG="${STARPRO_RUN_TAG:-llavaqr_anchorfloor_a2_64_20260724}"
LOG_ROOT="${STARPRO_LOG_ROOT:-$ROOT/results_aihc_qr_anchorfloor_a2_64_20260724}"
OUT="$LOG_ROOT/$TAG"
# arm = "scorer:anchorM"
ARMS="${STARPRO_ARMS:-qr:0 qr:8 qr:16 random:8}"
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

for arm in $ARMS; do
  scorer="${arm%%:*}"
  m="${arm##*:}"
  export STAGE1_SCORER="$scorer"
  export STAGE2_ANCHOR_M="$m"
  SUBOUT="$OUT/${scorer}_m${m}"
  mkdir -p "$SUBOUT"
  export RUN_TAG="${TAG}_${scorer}_m${m}"
  export RESULT_DIR="$SUBOUT"
  {
    echo "CONFIG tag=$TAG arm=$arm scorer=$scorer STAGE2_ANCHOR_M=$m METHOD=$METHOD TOKEN=$TOKEN STAGE1_MULT=$STAGE1_MULT TEXT_AGG_MODE=$TEXT_AGG_MODE STAR_KEEP_POSIDS=$STAR_KEEP_POSIDS STAR_CAUSAL_FIX=$STAR_CAUSAL_FIX"
    for bench in $BENCHES; do
      echo "===== BEGIN $TAG $arm $bench ====="
      bash "scripts/v1_5/7b/${bench}.sh" "$METHOD" "$TOKEN" 2>&1 | tee "$SUBOUT/${bench}.fulllog"
      echo "===== END $TAG $arm $bench ====="
    done
  } 2>&1 | tee "$SUBOUT/suite.log"
done

STARPRO_OUT="$OUT" STARPRO_ARMS="$ARMS" STARPRO_BENCHES="$BENCHES" python3 - <<'PY'
import os, re
root=os.environ["STARPRO_OUT"]; arms=os.environ["STARPRO_ARMS"].split(); benches=os.environ["STARPRO_BENCHES"].split()
pats={"gqa":r"Accuracy:\s*([\d.]+)%","textvqa":r"Accuracy:\s*([\d.]+)%","pope":r"Average F1 score:\s*([\d.]+)"}
bad=("Traceback","RuntimeError","CUDA out of memory","FileNotFoundError","[ERROR]")
res={}
print("=================== ANCHOR-FLOOR PAIRED RESULT (T=64, a2, average_all) ===================")
for arm in arms:
    scorer,m=arm.split(":"); key=scorer+"_m"+m; res[key]={}
    for bench in benches:
        p=os.path.join(root,key,bench+".fulllog")
        if not os.path.exists(p): print(key+"/"+bench+": MISSING_LOG"); continue
        t=open(p,errors="replace").read()
        hit=[w for w in bad if w in t]
        aa=t.count("Average-all (")
        if hit: print(key+"/"+bench+": ERROR "+str(hit)); continue
        mm=re.findall(pats[bench],t)
        if not mm: print(key+"/"+bench+": MISSING_SCORE avgall="+str(aa)); continue
        v=float(mm[-1]); v=v*100.0 if bench=="pope" else v
        res[key][bench]=v
        print("%s/%s: %.4f  (avgall_marker=%d)" % (key,bench,v,aa))
print("--------------------- vs baseline qr_m0 ---------------------")
base=res.get("qr_m0",{})
for key in res:
    if key=="qr_m0": continue
    parts=[]
    for bench in benches:
        a=res[key].get(bench); b=base.get(bench)
        if a is not None and b is not None: parts.append("%s d=%+.2f" % (bench,a-b))
    if parts: print(key+":  "+"  ".join(parts))
print("NOTE win = TextVQA up AND GQA/POPE not down; qr_m8 must also beat random_m8 (coverage-specific).")
print("=========================================================================================")
PY

echo "ALL_DONE llava_qr_anchorfloor_a2_64_20260724"
