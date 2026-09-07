#!/bin/bash
# 补齐 4B hide 无特权句臂: 405 题, HF 本地推理, 单卡 ~30min。
# 输出: eval/model_answer/treebench/qwen3.5-4b-priv-hide-nosent-nothink_answer.jsonl
#   sbatch 52_4b_hide_nosent.sbatch
#SBATCH --job-name=4b-hide-nosent
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/nosent4b-%j.log
#SBATCH --open-mode=append
set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
echo "job ${SLURM_JOB_ID:-NA} on $(hostname)  $(date)"
cd "$VOPD_ROOT/eval/treebench_probe"
"$VOPD_PY" infer_privilege.py --model Qwen/Qwen3.5-4B --privilege hide \
  --no-priv-sentence --no-think
rc=$?
echo "exit=$rc  $(date)"
exit $rc
