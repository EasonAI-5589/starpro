#!/bin/bash
# cd LLaVA-STAR-Pro2
# conda activate llava
# deactivate #fa_xf
export http_proxy=http://192.168.32.28:18000  
export https_proxy=http://192.168.32.28:18000 
# bash eval.sh

# ==================== Configuration ====================
# Model version: v1_5, v1_6
MODEL_VERSION="v1_5"

# Model scale: 7b, 13b
MODEL_SCALE="7b"

# Compression method: fastv, sparsevlm, pdrop, visionzip, dart, divprune, dp3, cdp3,
#   star_pro, star_v2, star_v2_anchor, star_v5, thcp, prefixvlm, prefixvlm_2,
#   mustdrop, prumerge, fastervlm, svdvlm, trim, scope, d2p, vscan, vanilla
METHOD="star_pro"

# Token budgets: 32, 64, 128, 192, 256
TOKEN_BUDGETS=(32 64 128)

# GPU devices
GPUS="0,1,2,3"

# Evaluation tasks: mme, pope, sqa, textvqa, gqa, vqav2, vizwiz, mmbench, mmbench_cn, mmvet
TASKS=(mmvet mmbench mmbench_cn mme pope sqa textvqa gqa)

# ==================== Run Evaluations ====================
for task in "${TASKS[@]}"; do
    for token_budget in "${TOKEN_BUDGETS[@]}"; do
        echo "=========================================="
        echo ">>> Running: $task with $METHOD (budget=$token_budget)"
        echo "=========================================="

        if [ "$METHOD" = "vscan" ]; then
            case "$token_budget" in
                128)
                    STAGE1=144
                    STAGE2=112
                    ;;
                64)
                    STAGE1=96
                    STAGE2=32
                    ;;
                32)
                    STAGE1=32
                    STAGE2=32
                    ;;
                *)
                    echo "[ERROR] Unsupported vscan budget: $token_budget"
                    echo "        Supported: 32, 64, 128"
                    exit 1
                    ;;
            esac
            echo ">>> VScan mapping: budget=${token_budget} -> s1=${STAGE1}, s2=${STAGE2}, avg=$(((STAGE1 + STAGE2)/2))"
            CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $STAGE1 $STAGE2
        else
            CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $token_budget
        fi
    done
done





# # ==================== Configuration ====================
# # Model version: v1_5, v1_6
# MODEL_VERSION="v1_6"

# # Model scale: 7b, 13b
# MODEL_SCALE="7b"

# # Compression method: fastv, sparsevlm, pdrop, visionzip, dart, divprune, dp3, cdp3, star, star_v2, thcp, vanilla
# METHOD="star_pro"

# # Token budgets: 
# TOKEN_BUDGETS=(160 320 640)

# # GPU devices
# GPUS="0,1,2,3,4,5,6,7"

# # Evaluation tasks: mme, pope, sqa, textvqa, gqa, vqav2, vizwiz, mmbench, mmbench_cn, mmvet
# TASKS=(mme)

# # ==================== Run Evaluations ====================
# for task in "${TASKS[@]}"; do
#     for token_budget in "${TOKEN_BUDGETS[@]}"; do
#         echo "=========================================="
#         echo ">>> Running: $task with $METHOD (budget=$token_budget)"
#         echo "=========================================="
#         CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $token_budget
#     done
# done


# # ==================== Configuration ====================
# # Model version: v1_5, v1_6
# MODEL_VERSION="v1_6"

# # Model scale: 7b, 13b
# MODEL_SCALE="13b"

# # Compression method: fastv, sparsevlm, pdrop, visionzip, dart, divprune, dp3, cdp3, star, star_v2, thcp, vanilla
# METHOD="star_pro"

# # Token budgets: 
# TOKEN_BUDGETS=(320)

# # GPU devices
# GPUS="0,1,2,3,4,5,6,7"

# # Evaluation tasks: mme, pope, sqa, textvqa, gqa, vqav2, vizwiz, mmbench, mmbench_cn, mmvet
# TASKS=(vqav2)

# # ==================== Run Evaluations ====================
# for task in "${TASKS[@]}"; do
#     for token_budget in "${TOKEN_BUDGETS[@]}"; do
#         echo "=========================================="
#         echo ">>> Running: $task with $METHOD (budget=$token_budget)"
#         echo "=========================================="
#         CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $token_budget
#     done
# done











# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/13b/mme.sh star_pro 128

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/pope.sh star_pro 128


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/sqa.sh star_pro 128

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh star_pro 64

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/vqav2.sh star_pro 32

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mmvet.sh star_pro 32



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/13b/pope.sh star_pro 128


# export CKPT_DIR=/mnt/eason_ckp/models                                                                                                                                
# export DATA_DIR=/mnt/eason_ckp/LLaVA-Eval                                                                                                                     
# export RESULT_DIR=/mnt/eason/STAR-Pro-LLaVA/results             
# export http_proxy=http://192.168.32.28:18000                                                                                              
# export https_proxy=http://192.168.32.28:18000 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh divprune 128

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh vanilla 576
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh fastv 128


# # mustdrop eval
# for token in 128 64; do
#     for task in mmbench mmbench_cn ; do
#         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/$task.sh mustdrop $token
#     done
# done

# # mustdrop 128
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/textvqa.sh mustdrop 128
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/pope.sh mustdrop 128
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mmbench.sh mustdrop 128

# # mustdrop 128
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/textvqa.sh mustdrop 128
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/pope.sh mustdrop 64
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mmbench.sh mustdrop 64

# # vscan 128                                                                                                                                                                    
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh vscan 144 112                                                                                                  
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/gqa.sh vscan 144 112                                                                                                  
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/textvqa.sh vscan 144 112                                                                                                  
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/pope.sh vscan 144 112                                                                                                  
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mmbench.sh vscan 144 112 


# # vscan 64                                                                                                                                                                     
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh vscan 96 32                                                                                                   
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/pope.sh vscan 96 32    
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/textvqa.sh vscan 96 32
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/gqa.sh vscan 96 32
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mmbench.sh vscan 96 32



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mmbench.sh vscan 144 112 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mmbench.sh vscan 96 32

# # ==================== SCOPE Evaluations ====================
# # SCOPE: Saliency-Coverage Oriented Token Pruning (NeurIPS 2025)
# # Reference: https://github.com/kinredon/SCOPE

# # SCOPE with 64 tokens
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh scope 64
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/pope.sh scope 64
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/sqa.sh scope 64

# # SCOPE with 128 tokens
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh scope 128
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/pope.sh scope 128

# # SCOPE hyperparameter tuning (via environment variables)
# # ALPHA: exponent for attention scores (default: 1.0)
# # COMBINED: 'multi' (multiplication) or 'add' (addition)
# # Note: Uses same env var names as official SCOPE implementation
# ALPHA=0.5 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh scope 64
# COMBINED=add CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh scope 64