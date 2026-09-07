#!/bin/bash
# 提交训练作业。**只负责资源申请和环境**, 训练逻辑全在 26_run_train.sh 里 ——
# 上一轮 09/15/20/21 之间复制粘贴出了一堆各自的 bug, 这次不重复任何一行。
#
#   sbatch 27_train.sbatch                          # 默认 xy_rb, 120 步
#   sbatch --export=ALL,VARIANT=xy 27_train.sbatch  # 换变体
#   sbatch --export=ALL,STEPS=200 27_train.sbatch   # 换步数
#
#SBATCH --job-name=vopd-train
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/vopd-%j.log
#SBATCH --open-mode=append
set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
mkdir -p "$SETUP/logs"

echo "=========================================================="
echo " job : ${SLURM_JOB_ID:-NA}  on $(hostname)   $(date)"
echo " 变体: ${VARIANT:-xy_rb}    步数: ${STEPS:-120}"
echo "=========================================================="

bash "$SETUP/26_run_train.sh"
rc=$?

echo
echo " MaxRSS: $(sstat -j "${SLURM_JOB_ID:-0}" --format=MaxRSS -a -n 2>/dev/null | tr -d ' ' | head -1)"
echo " ⚠️  MaxRSS 是整作业历史高水位, 只增不减 —— 中途读数永远是下界。"
echo "    上一轮 step33 读到 227GiB, 跑到 step260 实际是 481.7GiB(512G 上限的 94%)。"
echo " exit=$rc  $(date)"
exit $rc
