#!/bin/bash
# Qwen3-VL 4B/8B direct 的 no-system 协议重跑 (32B 走 interactive 卡)。
# 协议演化记录:
#   prefill <think>   -> 8B 直接 EOS (263/405 空回复), 废弃
#   system prompt     -> 32B 首 token 即 EOS (二分实锤), 废弃
#   标准协议 = 无预填 + 无 system, <answer> 指令在 user 后缀里, 三个规模均耐受
#SBATCH --job-name=q3vl-nosys
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/q3vl-nosys-%j.log
#SBATCH --open-mode=append

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
mkdir -p "$SETUP/logs"
OUT=$VOPD_ROOT/eval/model_answer/treebench

echo "================ $(hostname)  $(date) ================"
for M in 4B 8B; do
  echo "##### Qwen3-VL-$M nosys direct  $(date)"
  "$VOPD_PY" "$VOPD_ROOT/eval/treebench_probe/infer_staged.py" \
    --model "Qwen/Qwen3-VL-${M}-Instruct" --condition direct --no-prefill --no-system \
    --out "$OUT/qwen3-vl-${M,,}-instruct-direct-nosys_answer.jsonl"
  echo "##### rc=$?  $(date)"
done
echo " 全部结束 $(date)"
