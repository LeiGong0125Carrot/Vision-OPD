#!/bin/bash
# Aha + pair 特权对 + plausibility floor (α=0.1): 冠军骨架 × 防护件
#   pair TB 峰后回落 (52.35@35 → ~50), floor 的止漂特性对症; 机制正交性已由
#   center+floor 叠加臂验证。61 配方, 51 步 1 epoch。
#   sbatch opsa/aha/train_pairfloor.sbatch
#SBATCH --job-name=aha-pairfloor
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=10:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/aha-pairfloor-%j.log
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
AHA_FLOOR_ALPHA=0.1 \
TASK_TRAIN_FILE=/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA/data/TreeVGR-RL-37K/train_6karmA_pair.parquet \
EXPERIMENT_SUFFIX=pairfloor6k \
bash opsa/aha/run_aha_4b.sh
