#!/bin/bash
# think 模式组合特权: 4B draw_box (~4h) + 9B draw_box (~7h, 时限被切就 resume 重投)。
#SBATCH --job-name=q35-think-dbox
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/q35-think-dbox-%j.log
#SBATCH --open-mode=append

set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
PROBE="$VOPD_ROOT/eval/treebench_probe"; LABELS="$VOPD_ROOT/eval/treebench_gt_labels.jsonl"
echo "================ $(hostname)  $(date) ================"

echo "##### 4B think draw_box  $(date)"
"$VOPD_PY" "$PROBE/infer_privilege.py" --model Qwen/Qwen3.5-4B \
  --privilege draw_box --labels-from "$LABELS" --max-new-tokens 4096
echo "##### rc=$?  $(date)"

echo "##### 9B think draw_box  $(date)"
"$VOPD_PY" "$PROBE/infer_privilege.py" --model Qwen/Qwen3.5-9B \
  --privilege draw_box --labels-from "$LABELS" --max-new-tokens 4096
echo "##### rc=$?  $(date)"
echo " 全部结束 $(date)"
