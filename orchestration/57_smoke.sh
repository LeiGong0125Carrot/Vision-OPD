#!/bin/bash
# 单卡冒烟封装: bash 57_smoke.sh {v0-mixed|v0-hide|evtneg-hide|sav2-hide|sav2-mixed}
set -uo pipefail
ARM="${1:?用法: bash 57_smoke.sh v0-mixed|v0-hide|evtneg-hide|sav2-hide|sav2-mixed|v0-6k}"
VOPD=/sfs/weka/scratch/nkw3mr/Vision-OPD
DATA=$VOPD/data/TreeVGR-RL-37K

export TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 ROLLOUT_N=2
export TRAINER_N_GPUS_PER_NODE=1 TRAINER_SAVE_FREQ=-1 ROLLOUT_GPU_MEMORY_UTILIZATION=0.55
export EXPERIMENT_SUFFIX=smoke   # 冒烟专用目录, 避免撞上历史 checkpoint 触发 resume

case "$ARM" in
  v0-mixed)
    export SA_ENABLE=False EVT_ENABLE=False TASK_TRAIN_FILE=$DATA/train_sa4k_mixed.parquet ;;
  v0-hide)
    export SA_ENABLE=False EVT_ENABLE=False TASK_TRAIN_FILE=$DATA/train_sa4k_hide.parquet ;;
  v0-6k)
    # Arm A: Vision-OPD-6K 过滤池 (高清+框>=1%+非计数, 2459行), hide无句教师
    # 高清体制: 最大图 ≈6.1k 视觉token, 预算抬到 8192 (不降采样, 保持 TreeBench 体制)
    # 显存: 7k+ token 序列的全词表归约上量大, vLLM 常驻降 0.45 + 抗碎片腾余量
    export SA_ENABLE=False EVT_ENABLE=False TASK_TRAIN_FILE=$DATA/train_6k_armA_hide.parquet \
           MAX_PROMPT_LENGTH=8192 ROLLOUT_GPU_MEMORY_UTILIZATION=0.45 \
           PYTORCH_ALLOC_CONF=expandable_segments:True ACTIVATION_OFFLOAD=True ;;
  evtneg-hide)
    export EVT_ENABLE=True EVT_NEGATIVE_ONLY=True \
           EVT_UBAR_INIT=$DATA/evt_ubar_init_hide.json \
           TASK_TRAIN_FILE=$DATA/train_sa4k_hide.parquet ;;
  sav2-hide)
    export SA_ENABLE=True EVT_ENABLE=False TASK_TRAIN_FILE=$DATA/train_sa4k_hide.parquet ;;
  sav2-mixed)
    export SA_ENABLE=True EVT_ENABLE=False TASK_TRAIN_FILE=$DATA/train_sa4k_mixed.parquet ;;
  evtneg-6k)
    # EVT-neg × Arm A 数据 (6K高清过滤池): advantage 形式在高清体制下的对照冒烟
    export EVT_ENABLE=True EVT_NEGATIVE_ONLY=True \
           EVT_UBAR_INIT=$DATA/evt_ubar_init_hide.json \
           TASK_TRAIN_FILE=$DATA/train_6k_armA_hide.parquet \
           MAX_PROMPT_LENGTH=8192 ROLLOUT_GPU_MEMORY_UTILIZATION=0.45 \
           PYTORCH_ALLOC_CONF=expandable_segments:True ACTIVATION_OFFLOAD=False \
           TRAINER_N_GPUS_PER_NODE=2
    EXTRA="actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 actor_rollout_ref.ref.ulysses_sequence_parallel_size=2" ;;
  v0-6k-sp2)
    # Arm A 双卡序列并行冒烟: 每卡序列减半, 解 7k-token 序列的显存瓶颈
    export SA_ENABLE=False EVT_ENABLE=False TASK_TRAIN_FILE=$DATA/train_6k_armA_hide.parquet \
           MAX_PROMPT_LENGTH=8192 ROLLOUT_GPU_MEMORY_UTILIZATION=0.45 \
           PYTORCH_ALLOC_CONF=expandable_segments:True ACTIVATION_OFFLOAD=True \
           TRAINER_N_GPUS_PER_NODE=2
    EXTRA="actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 actor_rollout_ref.ref.ulysses_sequence_parallel_size=2" ;;
  *) echo "未知臂: $ARM"; exit 1 ;;
esac

echo ">>> 冒烟臂: $ARM  数据: $(basename "$TASK_TRAIN_FILE")"
cd "$VOPD"
exec bash scripts/run_state_adaptive_4b.sh \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  ${EXTRA:-}
