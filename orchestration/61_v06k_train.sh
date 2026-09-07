#!/bin/bash
# V0 × Arm A 先导 (interactive session, 2卡, 512G):
#   数据 = train_6k_armA_hide.parquet (6K高清过滤池 2459行, hide无句教师)
#   方法 = V0 标准 OPD (JSD α=0.5, top-100+尾桶, 冻结教师)
#   48题×2rollout=96条/步 (session CPU 安全线内), 51步全程 ~6h; 看健康指标可随时 Ctrl+C
#   后缀 6karmA-pre: 与未来 sbatch 正式跑 (96×8, 后缀 6karmA) 目录隔离
# 用法: [TRAINER_LOGGER='["console","tensorboard","wandb"]'] bash 61_v06k_train.sh
set -uo pipefail
VOPD=/sfs/weka/scratch/nkw3mr/Vision-OPD
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

cd "$VOPD"
exec bash scripts/run_state_adaptive_4b.sh
