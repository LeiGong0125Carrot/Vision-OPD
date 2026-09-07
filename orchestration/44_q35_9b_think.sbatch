#!/bin/bash
# Qwen3.5-9B think 两臂: direct 基线 (71 行已恢复续跑) + draw。~64s/题, 各 ~6-7h。
# 时限内跑不完就 resume 重投本文件。
#SBATCH --job-name=q35-9b-think
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/q35-9b-think-%j.log
#SBATCH --open-mode=append

set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
PROBE="$VOPD_ROOT/eval/treebench_probe"; LABELS="$VOPD_ROOT/eval/treebench_gt_labels.jsonl"
echo "================ $(hostname)  $(date) ================"

echo "##### 9B think direct (resume)  $(date)"
"$VOPD_PY" "$PROBE/infer_staged.py" --model Qwen/Qwen3.5-9B --condition direct \
  --no-prefill --no-system --max-new-tokens 4096
echo "##### rc=$?  $(date)"

echo "##### 9B think draw  $(date)"
"$VOPD_PY" "$PROBE/infer_privilege.py" --model Qwen/Qwen3.5-9B \
  --privilege draw --labels-from "$LABELS" --max-new-tokens 4096
echo "##### rc=$?  $(date)"
echo " 全部结束 $(date)"
