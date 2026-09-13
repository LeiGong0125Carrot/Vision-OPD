#!/bin/bash
# TZR 一体化流水线: 训练 (auto-resume 至 51 步) + LLM-as-judge 评测
#   训练 = 三区重建 (pair 数据, AHA_ZONE=True, 61 配方); 训练成功后原地接
#   合并 → TreeBench → V* → gpt-oss-120b 判卷 → 汇总 (eval_opsa.sbatch 复用)。
#   sbatch opsa/aha/train_eval_tzr.sbatch
#SBATCH --job-name=tzr-train-eval
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=14:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/tzr-train-eval-%j.log
set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
export TRITON_CACHE_DIR=/tmp/triton-$USER-aha
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$USER-aha
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True
CLONE=/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA
cd "$CLONE"

echo "========== [1/2] TZR 训练 (resume 自 $(cat checkpoints/Aha-zone-Qwen3.5-4B-tzr6k/latest_checkpointed_iteration.txt 2>/dev/null || echo 0) 步) $(date) =========="
TRAIN_BATCH_SIZE=48 PPO_MINI_BATCH_SIZE=48 ROLLOUT_N=2 \
MAX_PROMPT_LENGTH=8192 ULYSSES_SP=2 ACTIVATION_OFFLOAD=True \
ACTOR_USE_DYNAMIC_BSZ=False ACTOR_PPO_MICRO_BS=1 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.45 \
TRAINER_N_GPUS_PER_NODE=2 \
AHA_ZONE=True \
TASK_TRAIN_FILE=$CLONE/data/TreeVGR-RL-37K/train_6karmA_pair.parquet \
EXPERIMENT_SUFFIX=tzr6k \
bash opsa/aha/run_aha_4b.sh
TRAIN_RC=$?
FINAL_STEP=$(cat checkpoints/Aha-zone-Qwen3.5-4B-tzr6k/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
echo "训练退出码 $TRAIN_RC, 最终 checkpoint step $FINAL_STEP  $(date)"
if [ "$FINAL_STEP" -lt 51 ]; then
    echo "❌ 训练未到 51 步, 跳过评测 (重交本 sbatch 会自动续训)"; exit 1
fi

echo "========== [2/2] 评测 (judge 口径) $(date) =========="
# 清训练残留, 释放显存给 vLLM 推理与 judge 服务
pkill -u $USER -f "ray::" 2>/dev/null; pkill -u $USER -f "main_ppo" 2>/dev/null
pkill -u $USER -f "raylet" 2>/dev/null; pkill -u $USER -f "gcs_server" 2>/dev/null
sleep 15
export ARM_TAG=Aha-zone-Qwen3.5-4B-tzr6k ARM_SHORT=ahatzr6k
bash "$CLONE/opsa/eval_opsa.sbatch"
