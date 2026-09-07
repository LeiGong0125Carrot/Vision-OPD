#!/bin/bash
# HiDe 移植双变体: hide_compact(仅紧凑图, 替换式) + hide_full(原图+紧凑图, 原版二段协议)。
# 与 hide(绝对位置版, 19046037) 三方对打。GT 框替代注意力检测。
#SBATCH --job-name=q35-hidec
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/q35-hidec-%j.log
#SBATCH --open-mode=append
set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
PROBE="$VOPD_ROOT/eval/treebench_probe"
echo "================ $(hostname)  $(date) ================"
run(){ echo "##### $*  $(date)"; "$VOPD_PY" "$PROBE/infer_privilege.py" "$@"; echo "##### rc=$?  $(date)"; }
run --model Qwen/Qwen3.5-4B --privilege hide_compact --no-think
run --model Qwen/Qwen3.5-4B --privilege hide_full    --no-think
run --model Qwen/Qwen3.5-9B --privilege hide_compact --no-think
run --model Qwen/Qwen3.5-9B --privilege hide_full    --no-think
echo " 全部结束 $(date)"
