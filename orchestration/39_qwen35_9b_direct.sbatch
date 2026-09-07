#!/bin/bash
# Qwen3.5-9B direct 两臂 (nothink@1024 / think@4096) + 补欠的 Qwen3-VL-8B 盲+框臂。
# 4B 对照: nothink 46.42 / think 在 interactive 上跑。Qwen3-VL-8B 盲地板(none-blind)已完成。
# 9B 首跑自动下载权重 (~18GB, HF cache 在 scratch)。
#SBATCH --job-name=q35-9b-direct
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/q35-9b-direct-%j.log
#SBATCH --open-mode=append

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
mkdir -p "$SETUP/logs"
PROBE="$VOPD_ROOT/eval/treebench_probe"

echo "================ $(hostname)  $(date) ================"

echo "##### Qwen3.5-9B direct nothink  $(date)"
"$VOPD_PY" "$PROBE/infer_staged.py" --model Qwen/Qwen3.5-9B --condition direct \
  --no-prefill --no-system --no-think
echo "##### rc=$?  $(date)"

echo "##### Qwen3.5-9B direct think@4096  $(date)"
"$VOPD_PY" "$PROBE/infer_staged.py" --model Qwen/Qwen3.5-9B --condition direct \
  --no-prefill --no-system --max-new-tokens 4096
echo "##### rc=$?  $(date)"

echo "##### Qwen3-VL-8B blind+labeled-box (盲臂补欠)  $(date)"
"$VOPD_PY" "$PROBE/infer_privilege.py" --model Qwen/Qwen3-VL-8B-Instruct \
  --privilege box_block --labels-from "$VOPD_ROOT/eval/treebench_gt_labels.jsonl" --no-image
echo "##### rc=$?  $(date)"

echo " 全部结束 $(date)"
