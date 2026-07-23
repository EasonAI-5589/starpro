export http_proxy=http://192.168.32.28:18000  
export https_proxy=http://192.168.32.28:18000  
# bash select_red_boxes.sh

/mnt/eason/miniconda3/envs/llava/bin/python \
    /mnt/eason/LLaVA-STAR-Pro2/select_red_boxes.py \
    /mnt/eason/LLaVA-STAR-Pro2/mme_vis_output/position_clock_under_people_stage3_llm.png \
    --out /mnt/eason/LLaVA-STAR-Pro2/mme_vis_output/position_clock_under_people_stage3_llm_red_picked.png