#!/bin/bash
# 对 v1_181（The Kiss）三张 stage 图逐一进行红框标注
# grid 已预先计算: 21 rows x 28 cols（cell ~49x49px）
#
# 用法:
#   bash select_red_boxes_v1181.sh
#
# 每张图: 浏览器打开 http://<server_ip>:5000
#         点击保留的 token 格子 -> 变红
#         点 Save -> Ctrl+C 继续下一张

export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000

PYTHON=/mnt/eason/miniconda3/envs/llava/bin/python
SCRIPT=/mnt/eason/LLaVA-STAR-Pro2/select_red_boxes.py
DIR=/mnt/eason/LLaVA-STAR-Pro2/vis_mmvet_better_output/v1_181
PORT=5000
GRID="21x28"

IMAGES=(
    "${DIR}/01_stage1_arch.png"
    "${DIR}/02_stage2_llm.png"
    "${DIR}/03_stage3_llm.png"
)

for i in "${!IMAGES[@]}"; do
    IMG="${IMAGES[$i]}"
    OUT="${IMG%.png}_red.png"
    echo ""
    echo "[$((i+1))/3] $(basename $IMG)   grid=${GRID}"
    echo "  输出: $(basename $OUT)"
    echo "  >>> 浏览器打开 http://<server_ip>:${PORT}   完成后 Ctrl+C <<<"
    $PYTHON "$SCRIPT" "$IMG" --out "$OUT" --port $PORT --grid $GRID
done

echo ""
echo "完成！标注文件:"
ls -1 "${DIR}"/*_red.png 2>/dev/null
