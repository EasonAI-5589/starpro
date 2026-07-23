mmbench:
python scripts/eval_mmbench_local.py \
    --xlsx  /mnt/eason/LLaVA-STAR-Pro2/playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712/llava-v1.5-13b/prefixvlm_2/vtn_128.xlsx \
    --tsv   /mnt/eason_ckp/LLaVA-Eval/mmbench/mmbench_dev_20230712.tsv

mmbench_cn:
python scripts/eval_mmbench_local.py \
      --xlsx  /mnt/eason/LLaVA-STAR-Pro2/playground/data/eval/mmbench_cn/answers_upload/mmbench_dev_cn_20231003/llava-v1.5-13b/prefixvlm_2/vtn_128.xlsx \
      --tsv   /mnt/eason_ckp/LLaVA-Eval/mmbench_cn/mmbench_dev_cn_20231003.tsv

mme:                                                                                                                             
python playground/data/eval/MME/convert_answer_to_mme.py \                                                                                                   
    --data_path /mnt/eason_ckp/LLaVA-Eval/MME \                                                                                     
    --experiment llava_mme/lla va-v1.5-13b/prefixvlm_2/vtn_64/merge
python playground/data/eval/MME/eval_tool/calculation.py \                                                                                                             
      --results_dir answers/llava_mme/llava-v1.5-13b/prefixvlm_2/vtn_64/merge     

pope:
python llava/eval/eval_pope.py --annotation-dir /mnt/eason_ckp/LLaVA-Eval/pope/coco --question-file ./playground/data/eval/pope/llava_pope_test.jsonl --result-file ./playground/data/eval/pope/answers/llava_pope_test/llava-v1.5-13b/prefixvlm_2/vtn_32/merge.jsonl 

textvqa:
python llava/eval/eval_textvqa.py --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json --result-file ./playground/data/eval/textvqa/answers/llava_textvqa_val_v051_ocr/llava-v1.5-13b/prefixvlm_2/vtn_128/merge.jsonl

sqa:
python llava/eval/eval_science_qa.py --base-dir /mnt/eason_ckp/LLaVA-Eval/scienceqa --result-file ./playground/data/eval/scienceqa/answers/llava_test_CQM-I/llava-v1.5-13b/prefixvlm_2/vtn_64/merge.jsonl --output-file /tmp/sqa_results.json --output-result /tmp/sqa_output.json    

gqa:
python playground/data/eval/gqa/convert_predictions.py playground/data/eval/gqa/answers/llava_gqa_testdev_balanced/llava-v1.5-13b/prefixvlm_2/vtn_64/merge.jsonl /mnt/eason/testdev_balanced_predictions.json
python /mnt/eason_ckp/LLaVA-Eval/gqa/data/eval/eval.py --path /mnt/eason_ckp/LLaVA-Eval/gqa/data/questions --tier testdev_balanced --method /mnt/eason

mmvet:
none

