#!/bin/bash
# Qwen3.5-4B think 模式特权交互实验 (用户定稿: 先grounding再reasoning 是 think 原生行为,
# 视觉特权 x 原生 grounding 的交互是主实验; nothink 臂另跑做训练对齐对照)。
# ① think direct 基线 (246 行已恢复, 续 159 题) ② think draw (红框视觉特权)
# ③ think box_block labeled (文本坐标特权, 解耦用)
# 全部 max-new-tokens 4096; 若时限被切, resume 重投即可。
#SBATCH --job-name=q35-think-priv
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/q35-think-priv-%j.log
#SBATCH --open-mode=append

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
mkdir -p "$SETUP/logs"
PROBE="$VOPD_ROOT/eval/treebench_probe"
LABELS="$VOPD_ROOT/eval/treebench_gt_labels.jsonl"

echo "================ $(hostname)  $(date) ================"

echo "##### 1. Qwen3.5-4B think direct 基线 (resume 246->405)  $(date)"
"$VOPD_PY" "$PROBE/infer_staged.py" --model Qwen/Qwen3.5-4B --condition direct \
  --no-prefill --no-system --max-new-tokens 4096
echo "##### rc=$?  $(date)"

echo "##### 2. Qwen3.5-4B think draw labeled (视觉特权)  $(date)"
"$VOPD_PY" "$PROBE/infer_privilege.py" --model Qwen/Qwen3.5-4B \
  --privilege draw --labels-from "$LABELS" --max-new-tokens 4096
echo "##### rc=$?  $(date)"

echo "##### 3. Qwen3.5-4B think box_block labeled (文本特权)  $(date)"
"$VOPD_PY" "$PROBE/infer_privilege.py" --model Qwen/Qwen3.5-4B \
  --privilege box_block --labels-from "$LABELS" --max-new-tokens 4096
echo "##### rc=$?  $(date)"

echo " 全部结束 $(date)"
