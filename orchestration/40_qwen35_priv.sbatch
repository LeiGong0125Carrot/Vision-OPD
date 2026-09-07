#!/bin/bash
# Qwen3.5 特权窗口平移检验 (纯 nothink 版, 与训练模板对齐: 训练用空 think 块 = nothink)。
# think 臂全线取消 (2026-08-29 用户决定); 4B priv nothink 在 interactive 上并行跑, 不在此处。
# 基线: 3.5-4B nothink 46.42 / 3.5-9B nothink 48.89。
#SBATCH --job-name=q35-priv
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/q35-priv-%j.log
#SBATCH --open-mode=append

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
mkdir -p "$SETUP/logs"
PROBE="$VOPD_ROOT/eval/treebench_probe"
LABELS="$VOPD_ROOT/eval/treebench_gt_labels.jsonl"

echo "================ $(hostname)  $(date) ================"

echo "##### 1. Qwen3.5-9B priv labeled box_block nothink  $(date)"
"$VOPD_PY" "$PROBE/infer_privilege.py" --model Qwen/Qwen3.5-9B \
  --privilege box_block --labels-from "$LABELS" --no-think
echo "##### rc=$?  $(date)"

echo "##### 2. Qwen3-VL-8B blind+labeled-box  $(date)"
"$VOPD_PY" "$PROBE/infer_privilege.py" --model Qwen/Qwen3-VL-8B-Instruct \
  --privilege box_block --labels-from "$LABELS" --no-image
echo "##### rc=$?  $(date)"

echo " 全部结束 $(date)"
