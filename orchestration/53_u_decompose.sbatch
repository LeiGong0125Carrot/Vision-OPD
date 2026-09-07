#!/bin/bash
# u 分解探针: 训练分布上 region-vs-full 残差的 口味(DC)/证据(局域) 能量分解。
#   sbatch 53_u_decompose.sbatch
#SBATCH --job-name=u-decomp
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/udecomp-%j.log
#SBATCH --open-mode=append
set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
echo "job ${SLURM_JOB_ID:-NA} on $(hostname)  $(date)"
cd "$VOPD_ROOT/eval/treebench_probe"
"$VOPD_PY" probe_u_decompose.py --n 50
rc=$?
echo "exit=$rc  $(date)"
exit $rc
