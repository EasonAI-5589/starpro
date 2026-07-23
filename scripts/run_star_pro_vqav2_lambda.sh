#!/bin/bash
# Run STAR-Pro VQAv2 lambda ablation: λ ∈ {0.5, 1.0, 2.0} × T ∈ {128, 64, 32} (serial)
set -e

cd /mnt/eason/LLaVA-STAR-Pro

export CKPT_DIR=/mnt/eason_ckp/models
export DATA_DIR=/mnt/eason_ckp/LLaVA-Eval
export CUDA_VISIBLE_DEVICES=0,1,2,3

for LAMBDA_VAL in 0.5 1.0 2.0; do
    for VTN in 128 64 32; do
        echo "=========================================="
        echo "[$(date)] STAR-Pro VQAv2 lambda=${LAMBDA_VAL} vtn=${VTN}"
        echo "=========================================="
        LAMBDA=${LAMBDA_VAL} bash scripts/v1_5/7b/vqav2.sh star_pro ${VTN} 2>&1 | tee results/vqav2_scripts_v1_5_star_pro_lambda${LAMBDA_VAL}_vtn${VTN}.log
        
        # Backup submission files per lambda (vqav2.sh overwrites same path)
        UPLOAD_DIR="playground/data/eval/vqav2/answers_upload/llava_vqav2_mscoco_test-dev2015/llava-v1.5-7b/star_pro"
        ANSWERS_DIR="playground/data/eval/vqav2/answers/llava_vqav2_mscoco_test-dev2015/llava-v1.5-7b/star_pro"
        mkdir -p "${UPLOAD_DIR}/lambda_${LAMBDA_VAL}"
        cp "${UPLOAD_DIR}/vtn_${VTN}.json" "${UPLOAD_DIR}/lambda_${LAMBDA_VAL}/vtn_${VTN}.json" 2>/dev/null
        mkdir -p "${ANSWERS_DIR}/lambda_${LAMBDA_VAL}_vtn_${VTN}"
        cp "${ANSWERS_DIR}/vtn_${VTN}/merge.jsonl" "${ANSWERS_DIR}/lambda_${LAMBDA_VAL}_vtn_${VTN}/merge.jsonl" 2>/dev/null
        
        echo "[$(date)] Done lambda=${LAMBDA_VAL} vtn=${VTN} (backup saved)"
    done
done

echo ""
echo "All STAR-Pro VQAv2 lambda ablation runs complete!"
echo ""
