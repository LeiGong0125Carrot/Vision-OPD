#!/bin/bash
# Qwen3.5-27B 两臂 (direct nothink 基线 + labeled box_block nothink 特权) + 8B 盲+框补欠。
# 特权窗口上沿检验: 3.5 代 4B +2.72 已在窗口内; 27B 若如 3-VL-32B 一样无感/负,
# 倒 U 在新代际复现。27B 首跑自动下载 (~54GB)。9B priv 由用户在 interactive 手跑。
#SBATCH --job-name=q35-27b
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/q35-27b-%j.log
#SBATCH --open-mode=append

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
mkdir -p "$SETUP/logs"
PROBE="$VOPD_ROOT/eval/treebench_probe"
LABELS="$VOPD_ROOT/eval/treebench_gt_labels.jsonl"

echo "================ $(hostname)  $(date) ================"

echo "##### 1. Qwen3.5-27B direct nothink (基线)  $(date)"
"$VOPD_PY" "$PROBE/infer_staged.py" --model Qwen/Qwen3.5-27B --condition direct \
  --no-prefill --no-system --no-think
echo "##### rc=$?  $(date)"

echo "##### 2. Qwen3.5-27B priv labeled box_block nothink  $(date)"
"$VOPD_PY" "$PROBE/infer_privilege.py" --model Qwen/Qwen3.5-27B \
  --privilege box_block --labels-from "$LABELS" --no-think
echo "##### rc=$?  $(date)"

echo "##### 3. Qwen3-VL-8B blind+labeled-box (补欠)  $(date)"
"$VOPD_PY" "$PROBE/infer_privilege.py" --model Qwen/Qwen3-VL-8B-Instruct \
  --privilege box_block --labels-from "$LABELS" --no-image
echo "##### rc=$?  $(date)"

echo " 全部结束 $(date)"
