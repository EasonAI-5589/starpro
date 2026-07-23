
Step 1 — 先跑 vanilla（只跑一次）：         
cd /mnt/eason/LLaVA-STAR-Pro2           
python visualize_loss_diff.py --step collect_vanilla \
    --model_path /mnt/eason_ckp/models/llava-v1.5-7b \
    --n_samples 200 --output_dir ./loss_diff_results_mme_200

Step 2 — 每种方法独立跑（可以分开在不同 GPU 上）：
python visualize_loss_diff.py --step collect_method --method fastv    --budgets 32 64 128 --output_dir ./loss_diff_results_mme_200
python visualize_loss_diff.py --step collect_method --method divprune  --budgets 32 64 128 --output_dir ./loss_diff_results_mme_200
python visualize_loss_diff.py --step collect_method --method scope     --budgets 32 64 128 --output_dir ./loss_diff_results_mme_200
python visualize_loss_diff.py --step collect_method --method tops      --budgets 32 64 128 --output_dir ./loss_diff_results_mme_200

Step 2 — 每种方法独立跑（可以分开在不同 GPU 上）：
python visualize_loss_diff.py --step collect_method --method fastv    --budgets 32 64 128 --output_dir ./loss_diff_results
python visualize_loss_diff.py --step collect_method --method divprune  --budgets 32 64 128 --output_dir ./loss_diff_results
python visualize_loss_diff.py --step collect_method --method scope     --budgets 32 64 128 --output_dir ./loss_diff_results
python visualize_loss_diff.py --step collect_method --method tops      --budgets 32 64 128 --output_dir ./loss_diff_results

Step 2 — 每种方法独立跑（可以分开在不同 GPU 上）：
python visualize_loss_diff.py --step collect_method --method fastv    --budgets 32 64 128 --output_dir ./loss_diff_results
python visualize_loss_diff.py --step collect_method --method divprune  --budgets 32 64 128 --output_dir ./loss_diff_results
python visualize_loss_diff.py --step collect_method --method scope     --budgets 32 64 128 --output_dir ./loss_diff_results
python visualize_loss_diff.py --step collect_method --method tops      --budgets 32 64 128 --output_dir ./loss_diff_results

python visualize_loss_diff.py --step collect_method --method fastv    --budgets 32 64 128 --output_dir ./loss_diff_results
python visualize_loss_diff.py --step collect_method --method divprune  --budgets 32 64 128 --output_dir ./loss_diff_results
python visualize_loss_diff.py --step collect_method --method scope     --budgets 32 64 128 --output_dir ./loss_diff_results
python visualize_loss_diff.py --step collect_method --method tops      --budgets 32 64 128 --output_dir ./loss_diff_results


Step 3 — 画图：
python visualize_loss_diff.py --step plot --output_dir ./loss_diff_results

loss 的计算方式：vanilla 模型对每个样本的 top-1 预测 token 作为参考，loss = -log P(vanilla_top1 | question + image)，loss_diff = pruned_loss
- vanilla_loss，值越接近 0 越好。