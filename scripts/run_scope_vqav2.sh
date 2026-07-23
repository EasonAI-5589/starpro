#!/bin/bash
# Run SCOPE VQAv2 for all 3 token budgets (serial)
set -e

cd /mnt/eason/LLaVA-STAR-Pro

export CKPT_DIR=/mnt/eason_ckp/models
export DATA_DIR=/mnt/eason_ckp/LLaVA-Eval
export CUDA_VISIBLE_DEVICES=0,1,2,3

for VTN in 128 64 32; do
    echo "=========================================="
    echo "[$(date)] Starting SCOPE VQAv2 vtn=${VTN}"
    echo "=========================================="
    bash scripts/v1_5/7b/vqav2.sh scope ${VTN} 2>&1 | tee results/vqav2_scripts_v1_5_scope_vtn${VTN}.log
    echo "[$(date)] Done SCOPE VQAv2 vtn=${VTN}"
done

echo ""
echo "All SCOPE VQAv2 runs complete!"
echo ""
