#!/bin/bash
# V0 × Arm A 续训 (从 step15 自动断点续训到 51; verl resume_mode=auto 读 latest_checkpointed_iteration.txt)
# profile 与 61 先导完全一致 (48×2, sp2, 激活offload, 8192); 640G 防 CPU 缓升越线 (512G 实测在 step17-19 被杀)
#   sbatch 62_v06k_resume.sbatch
#SBATCH --job-name=v06k-resume
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=640G
#SBATCH --time=12:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/v06k-resume-%j.log
#SBATCH --open-mode=append
set -uo pipefail
VOPD=/sfs/weka/scratch/nkw3mr/Vision-OPD
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
DATA=$VOPD/data/TreeVGR-RL-37K

export TRAIN_BATCH_SIZE=48 PPO_MINI_BATCH_SIZE=48 ROLLOUT_N=2
export SA_ENABLE=False EVT_ENABLE=False
export TASK_TRAIN_FILE=$DATA/train_6k_armA_hide.parquet
export EXPERIMENT_SUFFIX=6karmA-pre
export MAX_PROMPT_LENGTH=8192 ULYSSES_SP=2 ACTIVATION_OFFLOAD=True
export ACTOR_USE_DYNAMIC_BSZ=False ACTOR_PPO_MICRO_BS=1
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRAINER_N_GPUS_PER_NODE=2

echo "== V0-6karmA 续训  job=${SLURM_JOB_ID:-NA}  节点=$(hostname)  $(date)"
echo "== 起点: step $(cat "$VOPD/checkpoints/StdOPD-region-Qwen3.5-4B-6karmA-pre/latest_checkpointed_iteration.txt" 2>/dev/null || echo '?')"
cd "$VOPD"
bash scripts/run_state_adaptive_4b.sh
rc=$?
echo "== 训练退出 rc=$rc  终点: step $(cat "$VOPD/checkpoints/StdOPD-region-Qwen3.5-4B-6karmA-pre/latest_checkpointed_iteration.txt" 2>/dev/null || echo '?')  $(date)"
exit $rc
