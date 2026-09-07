#!/bin/bash
# 完整 1 epoch 训练 (260 步)。与 27_train.sbatch 的唯一区别是**资源和存档间隔**,
# 训练逻辑仍然全在 26_run_train.sh 里, 一行都不复制。
#
#   sbatch 31_train_epoch.sbatch
#
# ─── 为什么是 260 步 ────────────────────────────────────────────────────────
# train.jsonl 6241 条 / train_batch_size 24 = 260.04 步 = 恰好一遍。
# 上一轮 120 步只跑了 46%。260 步内**每一步取到的题都是没训过的**(实测 120 步
# 里 2880 个 (step,题) 对去重后 2878 道, 跨 step 重复率 0.07%), 而 rollout dump
# 发生在用该 batch 更新之前 —— 所以整条 IoU 曲线本身就是流式 held-out 评测,
# 到 260 步为止这个性质都成立。**超过 260 步就不再成立了。**
#
# ─── 为什么 --mem 从 512G 提到 800G ─────────────────────────────────────────
# 上一轮(job 18871218, 120 步)的 RSS 是一路爬的阶梯: 216.7 -> 413.3 GiB,
# 收尾 MaxRSS 433 GiB(512G 的 85%), 而且到 step 117 还在涨。
# 阶梯的跳变点全部落在存档边界(19-21 / 39-41 / 59-61 / 79-81 / 99),
# 196 GiB 的总增长里约 186 GiB 来自存 checkpoint —— 每存一次泄漏 16~50 GiB。
# 横向对照: v2(260 步)在 step 33 是 227 GiB, 我们同一步是 273 GiB, 高 46 GiB;
# v2 跑到 step 260 是 481.7 GiB。按固定偏移推 260 步约 528 GiB, 按 step41->99
# 的斜率线性外推是 659 GiB —— **两种推法都超 512G**。
# 节点有 1500 GB, 800G 覆盖了较坏的那个外推, 且仍给同节点留 700G。
#
# ─── 为什么 SAVE_FREQ 从 20 提到 65 ─────────────────────────────────────────
# 既然泄漏是按"存档次数"而不是"步数"来的: 260 步 @20 要存 13 次, @65 只存 4 次
# (step 65/130/195/260)。同时省磁盘 —— 每个 checkpoint 54G, 13 次 = 702 GB,
# 4 次 = 216 GB(现有 checkpoints/ 已占 744G, 剩 11T, 两种都放得下但没必要)。
# 中间 checkpoint 不是主产出: IoU 曲线来自 rollout dump(每步都有), 与存档无关。
#
# ─── 为什么 --time 从 24h 提到 30h ──────────────────────────────────────────
# 上一轮步时是平的(step 1 预热 336s, 之后稳定 257~270s, 平均 269.4s),
# 260 x 269.4 = 19.5h。分区时限 3 天, 30h 给 1.5x 余量。
# **这个作业不能续跑** —— EMA teacher 不进 checkpoint(fsdp_workers.py:935 只存
# actor_module_fsdp), 续训会把它重置回底模, 必须一次跑完。
#
#SBATCH --job-name=vopd-e1
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=800G
#SBATCH --time=30:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/vopd-%j.log
#SBATCH --open-mode=append
set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
mkdir -p "$SETUP/logs"

# 全新一轮, 不续跑: TAG 换新 -> $CKPT 目录不存在 -> 不会命中 26_run_train.sh:156
# 的 resume 分支。沿用 vopd-xy_rb 会自动从 global_step_120 续上, 那样 EMA teacher
# 被重置回底模, 跑出来的既不是新实验也不是旧实验的延长。
export VARIANT="${VARIANT:-xy_rb}"
export STEPS="${STEPS:-260}"
export TAG="${TAG:-vopd-xy_rb-e1}"
export SAVE_FREQ="${SAVE_FREQ:-65}"

CKPT=/sfs/weka/scratch/nkw3mr/Vision-OPD/checkpoints/$TAG
if [ -f "$CKPT/latest_checkpointed_iteration.txt" ]; then
  echo "❌ $CKPT 已有 checkpoint (step $(cat "$CKPT/latest_checkpointed_iteration.txt"))。"
  echo "   继续跑会 resume, EMA teacher 会被重置。换个 TAG 或先删掉这个目录。"
  exit 1
fi

echo "=========================================================="
echo " job : ${SLURM_JOB_ID:-NA}  on $(hostname)   $(date)"
echo " 变体: $VARIANT   步数: $STEPS (=1 epoch)   存档间隔: $SAVE_FREQ"
echo " tag : $TAG       内存: 800G   时限: 30h"
echo "=========================================================="

bash "$SETUP/26_run_train.sh"
rc=$?

echo
echo " MaxRSS: $(sstat -j "${SLURM_JOB_ID:-0}" --format=MaxRSS -a -n 2>/dev/null | tr -d ' ' | head -1)"
echo " ⚠️  MaxRSS 是整作业历史高水位, 只增不减 —— 中途读数永远是下界。"
echo "    参考: 120 步那轮收尾 433 GiB; v2(260 步, save_freq=20) 481.7 GiB。"
echo " exit=$rc  $(date)"
exit $rc
