#!/bin/bash
# hide 形态 (HiDe-LPD 简化版, 用户提案): 黑底原位融合多 region 成单图。
# 对照: 4B region 50.12 / draw 51.60; 9B region 51.11 / draw 50.86。
# 判读: hide > region ⇒ 保空间关系有增量; hide > draw ⇒ 背景剥离比注意力引导更强。
#SBATCH --job-name=q35-hide
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/q35-hide-%j.log
#SBATCH --open-mode=append
set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
PROBE="$VOPD_ROOT/eval/treebench_probe"
echo "================ $(hostname)  $(date) ================"
run(){ echo "##### $*  $(date)"; "$VOPD_PY" "$PROBE/infer_privilege.py" "$@"; echo "##### rc=$?  $(date)"; }
run --model Qwen/Qwen3.5-4B --privilege hide --no-think
run --model Qwen/Qwen3.5-9B --privilege hide --no-think
echo " 全部结束 $(date)"
