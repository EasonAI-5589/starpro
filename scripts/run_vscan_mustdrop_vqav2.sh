#!/bin/bash
# Run VScan + MustDrop VQAv2 for comparison with SCOPE/STAR-Pro
# Serial execution as per Eason's rules
set -e

cd /mnt/eason/LLaVA-STAR-Pro

export CKPT_DIR=/mnt/eason_ckp/models
export DATA_DIR=/mnt/eason_ckp/LLaVA-Eval
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "============================================"
echo " VScan + MustDrop VQAv2 Comparison Runs"
echo " Started: $(date)"
echo "============================================"

# ─────────────────────────────────────────────
# Part 1: MustDrop (vtn=128, 64)
# Note: Only 64 and 128 have proper thresholds in MUSTDROP_THRESHOLDS
# ─────────────────────────────────────────────
for VTN in 128 64; do
    echo ""
    echo "=========================================="
    echo "[$(date)] Starting MustDrop VQAv2 vtn=${VTN}"
    echo "=========================================="
    bash scripts/v1_5/7b/vqav2.sh mustdrop ${VTN} 2>&1 | tee results/vqav2_scripts_v1_5_mustdrop_vtn${VTN}.log
    echo "[$(date)] Done MustDrop VQAv2 vtn=${VTN}"
done

# ─────────────────────────────────────────────
# Part 2: VScan (avg=128, 64, 32)
# VScan uses 2 stages: effective tokens = (stage1 + stage2) / 2
# stage2 default = 32
# avg=128 → stage1=224, stage2=32
# avg=64  → stage1=96,  stage2=32
# avg=32  → stage1=32,  stage2=32
# ─────────────────────────────────────────────
# VScan avg=128 (stage1=224, stage2=32)
echo ""
echo "=========================================="
echo "[$(date)] Starting VScan VQAv2 avg=128 (stage1=224, stage2=32)"
echo "=========================================="
bash scripts/v1_5/7b/vqav2.sh vscan 224 32 2>&1 | tee results/vqav2_scripts_v1_5_vscan_vtn128.log
echo "[$(date)] Done VScan VQAv2 avg=128"

# VScan avg=64 (stage1=96, stage2=32)
echo ""
echo "=========================================="
echo "[$(date)] Starting VScan VQAv2 avg=64 (stage1=96, stage2=32)"
echo "=========================================="
bash scripts/v1_5/7b/vqav2.sh vscan 96 32 2>&1 | tee results/vqav2_scripts_v1_5_vscan_vtn64.log
echo "[$(date)] Done VScan VQAv2 avg=64"

# VScan avg=32 (stage1=32, stage2=32)
echo ""
echo "=========================================="
echo "[$(date)] Starting VScan VQAv2 avg=32 (stage1=32, stage2=32)"
echo "=========================================="
bash scripts/v1_5/7b/vqav2.sh vscan 32 32 2>&1 | tee results/vqav2_scripts_v1_5_vscan_vtn32.log
echo "[$(date)] Done VScan VQAv2 avg=32"

echo ""
echo "============================================"
echo " All VScan + MustDrop VQAv2 runs complete!"
echo " Finished: $(date)"
echo "============================================"
echo ""
echo "Submission JSONs are in:"
echo "  playground/data/eval/vqav2/answers_upload/llava_vqav2_mscoco_test-dev2015/llava-v1.5-7b/"
echo "    mustdrop/vtn_128.json, mustdrop/vtn_64.json"
echo "    vscan/vtn_128.json, vscan/vtn_64.json, vscan/vtn_32.json"
