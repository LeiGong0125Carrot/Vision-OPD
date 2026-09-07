#!/bin/bash
# 16 格核心矩阵补缺 (1/2): 9B region nothink (~40min) + 4B region think (~4h)。
#SBATCH --job-name=q35-fill-a
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=7:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/q35-fill-a-%j.log
#SBATCH --open-mode=append
set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
PROBE="$VOPD_ROOT/eval/treebench_probe"
echo "================ $(hostname)  $(date) ================"
run(){ echo "##### $*  $(date)"; "$VOPD_PY" "$PROBE/infer_privilege.py" "$@"; echo "##### rc=$?  $(date)"; }
run --model Qwen/Qwen3.5-9B --privilege region --no-think
run --model Qwen/Qwen3.5-4B --privilege region --max-new-tokens 4096
echo " 全部结束 $(date)"
