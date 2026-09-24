#!/usr/bin/env bash
# 忠实复现冠军 Aha-Qwen3.5-4B-pair6k (OPSA-clone 代码), 唯一变量 = data.seed 被固定
# (冠军原为 data.seed=None 未设 → 非确定 shuffle; 这里固定以与 OPD-Aha 的 A/C 同种子对照)。
# 用法: [SEED=42] bash opsa/aha/train_pair_repro.sh
set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
export TRITON_CACHE_DIR=/tmp/triton-$USER-aharepro TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$USER-aharepro
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA
SEED="${SEED:-42}"

TRAIN_BATCH_SIZE=48 PPO_MINI_BATCH_SIZE=48 ROLLOUT_N=2 \
MAX_PROMPT_LENGTH=8192 ULYSSES_SP=2 ACTIVATION_OFFLOAD=True \
ACTOR_USE_DYNAMIC_BSZ=False ACTOR_PPO_MICRO_BS=1 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.45 TRAINER_N_GPUS_PER_NODE=2 \
TASK_TRAIN_FILE=/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA/data/TreeVGR-RL-37K/train_6karmA_pair.parquet \
EXPERIMENT_SUFFIX="pair6k_s${SEED}" \
bash opsa/aha/run_aha_4b.sh data.seed=${SEED}
# 注: AHA_FLOOR_ALPHA 不设 → null = 忠实 pair 复刻 (非 floor); AHA_BETA 默认 4.0。
# 对照: OPD-Aha 侧 A(pair_ahaA_6karmA, seed42, TB 49.63) / C(pair_coarse3, seed42, 49.88)。
# 若 OPSA(seed42) ≈ 49.63 → 无代码优势, 52.35 = shuffle 高抽; 若 ≈52 → 有残余代码差异。
