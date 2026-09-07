#!/bin/bash
# Qwen3-VL 4B/8B/32B 单轮先框后推 (boxfirst): 一次生成内 "JSON框 -> 推理 -> 答案"。
# 32B 冒烟已验证服从性: 10/10 出框 (norm1000, IoU 66.4), 10/10 有答案。
# 标准协议: 无预填 + 无 system; max_new_tokens 1536 (框要吃掉一部分预算)。
# direct 对照已齐: 4B 44.94 / 8B 44.94 / 32B 51.36 (TreeVGR-7B 49.63)。
#SBATCH --job-name=q3vl-boxfirst
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/q3vl-boxfirst-%j.log
#SBATCH --open-mode=append

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
mkdir -p "$SETUP/logs"

echo "================ $(hostname)  $(date) ================"
for M in 4B 8B 32B; do
  echo "##### Qwen3-VL-$M boxfirst  $(date)"
  "$VOPD_PY" "$VOPD_ROOT/eval/treebench_probe/infer_staged.py" \
    --model "Qwen/Qwen3-VL-${M}-Instruct" --condition boxfirst \
    --no-prefill --no-system --max-new-tokens 1536
  echo "##### rc=$?  $(date)"
done
echo " 全部结束 $(date)"
