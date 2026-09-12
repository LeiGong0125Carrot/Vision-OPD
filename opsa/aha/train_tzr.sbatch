#!/bin/bash
# TZR 三区重建: pair 骨架 + AGREE冻结/EVID抬升/PRIOR压制(双门)
#   对照裸 pair 52.35/93.72; 计划 TreeVGR/vlm_opsa_aha_plan.md
#
#   sbatch opsa/aha/train_pairfloor.sbatch
#SBATCH --job-name=aha-tzr
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=10:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/aha-tzr-%j.log
set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
export TRITON_CACHE_DIR=/tmp/triton-$USER-aha
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$USER-aha
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA

TRAIN_BATCH_SIZE=48 PPO_MINI_BATCH_SIZE=48 ROLLOUT_N=2 \
MAX_PROMPT_LENGTH=8192 ULYSSES_SP=2 ACTIVATION_OFFLOAD=True \
ACTOR_USE_DYNAMIC_BSZ=False ACTOR_PPO_MICRO_BS=1 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.45 \
TRAINER_N_GPUS_PER_NODE=2 \
AHA_ZONE=True \
TASK_TRAIN_FILE=/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA/data/TreeVGR-RL-37K/train_6karmA_pair.parquet \
EXPERIMENT_SUFFIX=tzr6k \
bash opsa/aha/run_aha_4b.sh
