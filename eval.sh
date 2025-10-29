CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/vqav2.sh cdp3_0_

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/gqa.sh cdp3_0_

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/7b/vizwiz.sh cdp3_0_

CUDA_VISIBLE_DEVICES=1 bash scripts/v1_5/7b/sqa.sh cdp3_0_

CUDA_VISIBLE_DEVICES=2 bash scripts/v1_5/7b/textvqa.sh cdp3_0_

CUDA_VISIBLE_DEVICES=3 bash scripts/v1_5/7b/pope.sh cdp3_0_

CUDA_VISIBLE_DEVICES=4 bash scripts/v1_5/7b/mme.sh cdp3_0_

CUDA_VISIBLE_DEVICES=5 bash scripts/v1_5/7b/mmbench.sh cdp3_0_

CUDA_VISIBLE_DEVICES=6 bash scripts/v1_5/7b/mmbench_cn.sh cdp3_0_

CUDA_VISIBLE_DEVICES=7 bash scripts/v1_5/7b/mmvet.sh cdp3_0_

#!/bin/bash

# ========== 配置 ==========
# 模型版本 (可选: v1_5, v1_6)
MODEL_VERSION="v1_5"

# 模型规模 (可选: 7b, 13b)
MODEL_SCALE="7b"

# 压缩方法 (可选: fastv, sparsevlm, pdrop, visionzip, dart, divprune, dp3, cdp3, star)
METHOD="star"

# Token预算 (可选: 64, 128, 192, 256等)
TOKEN_BUDGET=64

# GPU设备编号
GPUS="0,1,2,3,4,5,6,7"

# 评测任务 (可选: vqav2, gqa, textvqa, vizwiz, scienceqa, pope, mme, mmbench 等)
TASKS=(mme)

# ========== 运行 ==========
for task in "${TASKS[@]}"; do
    echo ">>> 运行: $task with $METHOD (budget=$TOKEN_BUDGET)"
    CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $TOKEN_BUDGET
done


# ========== 配置 ==========
# 模型版本 (可选: v1_5, v1_6)
MODEL_VERSION="v1_5"

# 模型规模 (可选: 7b, 13b)
MODEL_SCALE="7b"

# 压缩方法 (可选: fastv, sparsevlm, pdrop, visionzip, dart, divprune, dp3, cdp3, star, vani, thcp)
METHOD="thcp"

# Token预算 (可选: 64, 128, 192, 256等)
TOKEN_BUDGET=64

# GPU设备编号
GPUS="0,1,2,3,4,5,6,7"

# 评测任务 (可选: vqav2, gqa, textvqa, vizwiz, scienceqa, pope, mme, mmbench 等)
TASKS=(pope)

# ========== 运行 ==========
for task in "${TASKS[@]}"; do
    echo ">>> 运行: $task with $METHOD (budget=$TOKEN_BUDGET)"
    CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $TOKEN_BUDGET
done





# ========== 配置 ==========
# 模型版本 (可选: v1_5, v1_6)
MODEL_VERSION="v1_5"

# 模型规模 (可选: 7b, 13b)
MODEL_SCALE="7b"

# 压缩方法 (可选: fastv, sparsevlm, pdrop, visionzip, dart, divprune, dp3, cdp3, star, vani, thcp)
METHOD="dart"

# Token预算 (可选: 64, 128, 192, 256等)
TOKEN_BUDGET=64

# GPU设备编号
GPUS="0,1,2,3,4,5,6,7"

# 评测任务 (可选: vqav2, gqa, textvqa, vizwiz, scienceqa, pope, mme, mmbench 等)
TASKS=(mme)

# ========== 运行 ==========
for task in "${TASKS[@]}"; do
    echo ">>> 运行: $task with $METHOD (budget=$TOKEN_BUDGET)"
    CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $TOKEN_BUDGET
done

# ========== 配置 ==========
# 模型版本 (可选: v1_5, v1_6)
MODEL_VERSION="v1_5"

# 模型规模 (可选: 7b, 13b)
MODEL_SCALE="7b"

# 压缩方法 (可选: fastv, sparsevlm, pdrop, visionzip, dart, divprune, dp3, cdp3, star, vani, thcp)
METHOD="thcp"

# Token预算 (可选: 32, 64, 128, 192, 256等)
TOKEN_BUDGET=128

# GPU设备编号
GPUS="0,1,2,3,4,5,6,7"

# 评测任务 (可选: mme pope sqa textvqa gqa vqav2 vizwiz mmbench mmbench_cn mmvet等)
TASKS=(mme pope sqa textvqa gqa vqav2 vizwiz mmbench mmbench_cn mmvet)   

# ========== 运行 ==========
for task in "${TASKS[@]}"; do
    echo ">>> 运行: $task with $METHOD (budget=$TOKEN_BUDGET)"
    CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $TOKEN_BUDGET
done


# ========== 配置 ==========
# 模型版本 (可选: v1_5, v1_6)
MODEL_VERSION="v1_5"

# 模型规模 (可选: 7b, 13b)
MODEL_SCALE="13b"

# 压缩方法 (可选: fastv, sparsevlm, pdrop, visionzip, dart, divprune, dp3, cdp3, star, vani, thcp)
METHOD="thcp"

# Token预算 (可选: 32, 64, 128等)
TOKEN_BUDGET=128

# GPU设备编号
GPUS="0,1,2,3,4,5,6,7"

# 评测任务 (可选: mme pope sqa textvqa gqa vqav2 vizwiz mmbench mmbench_cn mmvet等)
TASKS=(mme pope sqa textvqa gqa vqav2 vizwiz mmbench mmbench_cn mmvet)   

# ========== 运行 ==========
for task in "${TASKS[@]}"; do
    echo ">>> 运行: $task with $METHOD (budget=$TOKEN_BUDGET)"
    CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $TOKEN_BUDGET
done










# ========== 配置 ==========
# 模型版本 (可选: v1_5, v1_6)
MODEL_VERSION="v1_5"

# 模型规模 (可选: 7b, 13b)
MODEL_SCALE="13b"

# 压缩方法 (可选: fastv, sparsevlm, pdrop, visionzip, dart, divprune, dp3, cdp3, star, vani, thcp)
METHOD="thcp"

# Token预算列表 (可选: 32, 64, 128, 192, 256等)
TOKEN_BUDGETS=(32 64 128)

# GPU设备编号
GPUS="0,1,2,3,4,5,6,7"

# 评测任务 (可选: mme pope sqa textvqa gqa vqav2 vizwiz mmbench mmbench_cn mmvet等)
TASKS=(mme pope sqa textvqa gqa vqav2 vizwiz mmbench mmbench_cn mmvet)   

# ========== 运行 ==========
for task in "${TASKS[@]}"; do
    for token_budget in "${TOKEN_BUDGETS[@]}"; do
        echo "=========================================="
        echo ">>> 运行: $task with $METHOD (budget=$token_budget)"
        echo "=========================================="
        CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $token_budget
    done
done











# ========== 配置 ==========
# 模型版本 (可选: v1_5, v1_6)
MODEL_VERSION="v1_5"

# 模型规模 (可选: 7b, 13b)
MODEL_SCALE="13b"

# 压缩方法 (可选: fastv, sparsevlm, pdrop, visionzip, dart, divprune, dp3, cdp3, star, vani, thcp)
METHOD="thcp"

# Token预算列表 (可选: 32, 64, 128, 192, 256等)
TOKEN_BUDGETS=(128 64 32)

# GPU设备编号
GPUS="0,1,2,3,4,5,6,7"

# 评测任务 (可选: mme pope sqa textvqa gqa vqav2 vizwiz mmbench mmbench_cn mmvet等)
TASKS=(vqav2 vizwiz)   

# ========== 运行 ==========
for task in "${TASKS[@]}"; do
    for token_budget in "${TOKEN_BUDGETS[@]}"; do
        echo "=========================================="
        echo ">>> 运行: $task with $METHOD (budget=$token_budget)"
        echo "=========================================="
        CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $token_budget
    done
done











# ========== 配置 ==========
# 模型版本 (可选: v1_5, v1_6)
MODEL_VERSION="v1_6"

# 模型规模 (可选: 7b, 13b)
MODEL_SCALE="7b"

# 压缩方法 (可选: fastv, sparsevlm, pdrop, visionzip, dart, divprune, dp3, cdp3, star, vani, thcp)
METHOD="thcp"

# Token预算列表 (可选: 32, 64, 128, 192, 256等)
TOKEN_BUDGETS=(128 64 32)

# GPU设备编号
GPUS="0,1,2,3,4,5,6,7"

# 评测任务 (可选: mme pope sqa textvqa gqa vqav2 vizwiz mmbench mmbench_cn mmvet等)
TASKS=(vqav2 vizwiz mmbench mmbench_cn mmvet)   

# ========== 运行 ==========
for task in "${TASKS[@]}"; do
    for token_budget in "${TOKEN_BUDGETS[@]}"; do
        echo "=========================================="
        echo ">>> 运行: $task with $METHOD (budget=$token_budget)"
        echo "=========================================="
        CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $token_budget
    done
done