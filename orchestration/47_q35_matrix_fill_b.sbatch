#!/bin/bash
# 16 格核心矩阵补缺 (2/2): 9B box_block think + 9B region think (各 ~6-7h, 切了重投续跑)。
#SBATCH --job-name=q35-fill-b
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/q35-fill-b-%j.log
#SBATCH --open-mode=append
set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
PROBE="$VOPD_ROOT/eval/treebench_probe"; LABELS="$VOPD_ROOT/eval/treebench_gt_labels.jsonl"
echo "================ $(hostname)  $(date) ================"
run(){ echo "##### $*  $(date)"; "$VOPD_PY" "$PROBE/infer_privilege.py" "$@"; echo "##### rc=$?  $(date)"; }
run --model Qwen/Qwen3.5-9B --privilege box_block --labels-from "$LABELS" --max-new-tokens 4096
run --model Qwen/Qwen3.5-9B --privilege region --max-new-tokens 4096
echo " 全部结束 $(date)"
