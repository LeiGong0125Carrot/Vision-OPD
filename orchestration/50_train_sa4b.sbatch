#!/bin/bash
# State-Adaptive OPD-Aha 4B pilot (双臂共用, 训练逻辑全在 scripts/run_state_adaptive_4b.sh)。
#
#   SA 臂 (方法):    sbatch 50_train_sa4b.sbatch
#   V0 臂 (标准OPD): sbatch -J v0opd4b --export=ALL,SA_ENABLE=False 50_train_sa4b.sbatch
#   3 卡时:          sbatch --gres=gpu:rtx_pro_6000:3 50_train_sa4b.sbatch
#                    (卡数由 SLURM_GPUS_ON_NODE 自动读取, batch 96 被 2/3 都整除)
#
# ─── 规模推算 ───────────────────────────────────────────────────────────────
# train_sa4k 4000 条 / batch 96 = 42 步 = 1 epoch(总步数由 total_epochs=1 自然截止)。
# e1 (batch 24 x n8, 2卡) 步时 ~270s, 本轮 tokens/步相当 (768 短序列 vs 192 长序列),
# SA 臂多一次 no-grad 教师前向 (+~10%): 估 42 x ~300s ≈ 3.5h, --time=12h 给足余量。
#
# ─── 内存: 600G, SAVE_FREQ=14 ──────────────────────────────────────────────
# e1 教训: 每存一次 checkpoint 泄漏 16~50 GiB CPU 内存(31_train_epoch.sbatch 注释)。
# 42 步 @14 只存 3 次 (step 14/28/42): 基线 ~220G + 3x50G = ~370G, 600G 覆盖。
#
# ─── 本轮可以续跑(与 e1 不同)────────────────────────────────────────────
# e1 的 EMA teacher 不进 checkpoint, 续训会重置。本轮 teacher_update_rate=0,
# teacher ≡ ref module ≡ 每次启动都从底模重新加载的冻结初始权重 —— resume 语义正确。
# 但两臂 checkpoint 目录 (SA-OPD-* / StdOPD-region-*) 自动分开, 不会互踩。
#
#SBATCH --job-name=saopd4b
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/saopd-%j.log
#SBATCH --open-mode=append
set -uo pipefail

REPO=/sfs/weka/scratch/nkw3mr/Vision-OPD
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh

export SA_ENABLE="${SA_ENABLE:-True}"
export TRAINER_N_GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-2}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-14}"

ARM_NAME=$([[ "$SA_ENABLE" == "True" ]] && echo "SA-OPD-Qwen3.5-4B" || echo "StdOPD-region-Qwen3.5-4B")
echo "=========================================================="
echo " job : ${SLURM_JOB_ID:-NA}  on $(hostname)   $(date)"
echo " 臂  : $ARM_NAME  (SA_ENABLE=$SA_ENABLE)"
echo " 卡数: $TRAINER_N_GPUS_PER_NODE   vLLM util: $ROLLOUT_GPU_MEMORY_UTILIZATION   save_freq: $TRAINER_SAVE_FREQ"
echo "=========================================================="

cd "$REPO"
bash scripts/run_state_adaptive_4b.sh "$@"
rc=$?

echo
echo " MaxRSS: $(sstat -j "${SLURM_JOB_ID:-0}" --format=MaxRSS -a -n 2>/dev/null | tr -d ' ' | head -1)"
echo " exit=$rc  $(date)"
exit $rc
