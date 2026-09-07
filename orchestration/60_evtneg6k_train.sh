#!/bin/bash
# EVT-neg × Arm A 全量训练 (interactive session, 2卡):
#   数据 = train_6k_armA_hide.parquet (6K高清过滤池 2459行, hide无句教师)
#   方法 = EVT-neg (advantage 形式, Â=min(ũ,0), 只压制不奖励)
#   显存/并行参数与 v0-6k-sp2 冒烟验证的一套完全一致
# checkpoint -> checkpoints/EVTneg-Qwen3.5-4B-6karmA (与 sbatch 的 V0 臂目录不冲突)
# 用法: bash 60_evtneg6k_train.sh
set -uo pipefail
VOPD=/sfs/weka/scratch/nkw3mr/Vision-OPD
DATA=$VOPD/data/TreeVGR-RL-37K

# 高清像素张量随 batch×rollout_n 线性顶爆 driver CPU (三次 cgroup OOM-kill 实测):
#   默认 n=8 时 48×8=384 条/步; 收敛到冒烟验证过的 n=2 → 96 条/步
export TRAIN_BATCH_SIZE=48 PPO_MINI_BATCH_SIZE=48 ROLLOUT_N=2
export SA_ENABLE=False EVT_ENABLE=True EVT_NEGATIVE_ONLY=True
export EVT_UBAR_INIT=$DATA/evt_ubar_init_hide.json
export TASK_TRAIN_FILE=$DATA/train_6k_armA_hide.parquet
export EXPERIMENT_SUFFIX=6karmA
# 激活 offload 关闭: EVT 无 top-100 logsumexp 机器, GPU 有余量; batch 96 下 offload 会打爆 512G CPU (实测 OOM-kill)
export MAX_PROMPT_LENGTH=8192 ULYSSES_SP=2 ACTIVATION_OFFLOAD=False
export ACTOR_USE_DYNAMIC_BSZ=False ACTOR_PPO_MICRO_BS=1
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRAINER_N_GPUS_PER_NODE=2

cd "$VOPD"
exec bash scripts/run_state_adaptive_4b.sh
