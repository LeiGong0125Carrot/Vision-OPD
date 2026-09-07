#!/bin/bash
# 基线: 4B 46.42 / 9B 48.89; 已测 4B: box 49.14 / draw 51.60 / crop 51.36。
#SBATCH --job-name=q35-nt-batch
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=7:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/q35-nt-batch-%j.log
#SBATCH --open-mode=append

set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
PROBE="$VOPD_ROOT/eval/treebench_probe"; LABELS="$VOPD_ROOT/eval/treebench_gt_labels.jsonl"
echo "================ $(hostname)  $(date) ================"
run(){ echo "##### $*  $(date)"; "$VOPD_PY" "$PROBE/infer_privilege.py" "$@"; echo "##### rc=$?  $(date)"; }

run --model Qwen/Qwen3.5-4B --privilege draw_box  --labels-from "$LABELS" --no-think
run --model Qwen/Qwen3.5-4B --privilege region    --no-think
run --model Qwen/Qwen3.5-4B --privilege none --with-hint --no-think
run --model Qwen/Qwen3.5-9B --privilege draw      --labels-from "$LABELS" --no-think
run --model Qwen/Qwen3.5-9B --privilege draw_box  --labels-from "$LABELS" --no-think
echo " 全部结束 $(date)"
