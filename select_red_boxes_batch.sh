#!/bin/bash
# 批量对 vis_mmvet_better_output 里的剪枝图进行手动红框标注
#
# 用法:
#   bash select_red_boxes_batch.sh
#
# 每张图依次启动 Flask 交互工具:
#   1. 浏览器打开 http://<server_ip>:5000
#   2. 点击要标红的 token 格子
#   3. 点 Save 保存（输出到原文件同目录，后缀 _red.png）
#   4. Ctrl+C 结束当前图，自动跳到下一张

export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000

PYTHON=/mnt/eason/miniconda3/envs/llava/bin/python
SCRIPT=/mnt/eason/LLaVA-STAR-Pro2/select_red_boxes.py
VIS_DIR=/mnt/eason/LLaVA-STAR-Pro2/vis_mmvet_better_output
PORT=5000

# 只标注 stage 图（跳过原图）
IMAGES=($(find "$VIS_DIR" -name "0[123]_stage*.png" | sort))

TOTAL=${#IMAGES[@]}
echo "共找到 ${TOTAL} 张 stage 图需要标注"
echo "每张图: 浏览器开 http://<server_ip>:${PORT}  →  点红框  →  Save  →  Ctrl+C 继续"
echo "=================================================="

for i in "${!IMAGES[@]}"; do
    IMG="${IMAGES[$i]}"
    OUT="${IMG%.png}_red.png"
    echo ""
    echo "[$((i+1))/${TOTAL}] $(basename $(dirname $IMG))/$(basename $IMG)"
    echo "  输出: $(basename $OUT)"
    echo "  >>> 浏览器打开 http://<server_ip>:${PORT}  完成后 Ctrl+C <<<"
    $PYTHON "$SCRIPT" "$IMG" --out "$OUT" --port $PORT
    echo "  已保存: $OUT"
done

echo ""
echo "=================================================="
echo "全部完成！标注结果保存在各子文件夹的 *_red.png 文件中"
