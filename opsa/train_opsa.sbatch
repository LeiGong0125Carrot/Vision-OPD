#!/bin/bash
# 视觉域 OPSA 正式训练 (Vision-OPD-OPSA clone, opsa 分支)
#   6K armA 高清 2459 题, 零监督零特权, 3 epochs ≈ 115 步, ~156s/步 ≈ 5h
#   资源 profile 抄 62_v06k_resume 先例: 640G (512G 实测被杀), rollout util 0.45
#   断点续训: verl resume_mode=auto 读 latest_checkpointed_iteration.txt, 重复 sbatch 即续
#   用法:  sbatch opsa/train_opsa.sbatch
#   干跑:  sbatch --export=ALL,TOTAL_TRAIN_STEPS=2,EXPERIMENT_SUFFIX=dryrun opsa/train_opsa.sbatch
#SBATCH --job-name=opsa-4b
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=640G
#SBATCH --time=12:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/opsa4b-%j.log
#SBATCH --open-mode=append
set -uo pipefail
CLONE=/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"

export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRAINER_N_GPUS_PER_NODE=2
# wandb 开启; 节点可联网, 若未 wandb login 则取消下行注释转离线 (事后 wandb sync)
# export WANDB_MODE=offline
export TRAINER_LOGGER='["console","tensorboard","wandb"]'

echo "== OPSA-4B 6karmA  job=${SLURM_JOB_ID:-NA}  节点=$(hostname)  $(date)"
echo "== 起点: step $(cat "$CLONE/checkpoints/OPSA-Qwen3.5-4B${EXPERIMENT_SUFFIX:+-$EXPERIMENT_SUFFIX}/latest_checkpointed_iteration.txt" 2>/dev/null || echo '0 (from scratch)')"
cd "$CLONE"
bash opsa/run_opsa_4b.sh
rc=$?
echo "== 训练退出 rc=$rc  终点: step $(cat "$CLONE/checkpoints/OPSA-Qwen3.5-4B${EXPERIMENT_SUFFIX:+-$EXPERIMENT_SUFFIX}/latest_checkpointed_iteration.txt" 2>/dev/null || echo '?')  $(date)"
exit $rc
