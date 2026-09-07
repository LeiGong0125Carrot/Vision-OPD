#!/bin/bash
# Vision-OPD 「语言特权」版全量训练 —— 1 epoch, 2 × RTX PRO 6000
#
#   sbatch 13_train_full.sbatch
#
# 实验内容: teacher 和 student 都看同一张干净全图, teacher 的唯一优势是文本里给出
# 的 bbox 坐标([0,1000] 归一化)。数据由 12_prep_data_bbox_text.py 生成, verl 代码
# 一行未改(走 ray_trainer.py:1300 现成的 teacher_prompt 通路)。
#
# 断点续训: trainer.resume_mode 默认是 auto(ppo_trainer.yaml:172), 配合下面独立的
# default_local_dir, 重新 sbatch 一次就会从自己最后一个 checkpoint 接着跑。
#
#SBATCH --job-name=vopd-bboxtext
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/vopd-bboxtext-%j.log
#SBATCH --open-mode=append

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
mkdir -p "$SETUP/logs"

echo "=========================================================="
echo " job      : ${SLURM_JOB_ID:-NA} on $(hostname)"
echo " start    : $(date)"
echo "=========================================================="

source "$SETUP/00_env.sh"

# ---------------------------------------------------------------------------
# 资源参数 —— 全部来自 2026-08-22 在 2 × RTX PRO 6000 上的实测, 不要凭感觉改
# ---------------------------------------------------------------------------
#   BATCH=24 RN=8            -> 192 序列/步; 6241 ÷ 24 = 260 步 = 1 epoch
#   稳态步时 257s (step 2/3 实测 259.34 / 255.28)
#     -> 260 × 257s ≈ 18.6h, 申请 24h 留约 29% 余量
#   MaxRSS 收敛在 387 GiB (每步增量 18.5 → 3.8 GiB, 已平台化) -> --mem=512G
#   显存 约 58 GiB/卡 (共 95.6) -> ROLLOUT_GPU_MEM=0.45
#   ppo_max_token_len_per_gpu 保持 9216: 试过 18432, 激活值翻倍后差 1.94 GiB OOM
export TRAIN_FILE="$VOPD_ROOT/data/train_bbox_text.parquet"
export BATCH=24
export RN=8
export ROLLOUT_GPU_MEM=0.45
export NGPUS=2
export STEPS=260                      # = trainer.total_training_steps

# checkpoint 用**独立目录**: 默认目录 checkpoints/Vision-OPD-Qwen3.5-4B 里有冒烟
# 测试留下的 global_step_2/3/4, 而 resume_mode=auto 会无差别地续上去。换个目录后
# 断点续训只会看到这次跑自己的 checkpoint, 语义干净。
CKPT_DIR="$VOPD_ROOT/checkpoints/vopd-bboxtext-2gpu"
ROLLOUT_DIR="$VOPD_ROOT/rollouts/vopd-bboxtext-2gpu"
# run_vision_opd.sh:59 只对它自己算出的路径做 mkdir, 我们用 hydra 覆盖后要自己建
mkdir -p "$CKPT_DIR" "$ROLLOUT_DIR"

# ---------------------------------------------------------------------------
# 前置检查 —— 早失败好过跑 5 分钟才炸
# ---------------------------------------------------------------------------
fail() { echo "❌ $*"; exit 1; }

[ -f "$TRAIN_FILE" ] || fail "训练数据不存在: $TRAIN_FILE (先跑 12_prep_data_bbox_text.py)"

n=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU")
[ "$n" = "$NGPUS" ] || fail "拿到 $n 张卡, 但配置按 $NGPUS 张调的。显存/步时都会不对。"

# flash-attn 是单 arch 编译的(无 PTX), 排到别的卡型会在训练中途报
# "no kernel image is available for execution on the device" —— 已经踩过一次。
want=$(cat "$VOPD_ENV/.flash_attn_archs" 2>/dev/null)
have=$(cuobjdump --list-elf "$VOPD_ENV"/lib/python3.12/site-packages/flash_attn_2_cuda*.so 2>/dev/null \
       | grep -oE "sm_[0-9]+" | sort -u | tr '\n' ' ')
echo ">>> GPU        : $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
echo ">>> flash-attn : 编译为 [$have], 期望含 sm_$want"
case "$have" in
  *"sm_$want"*) ;;
  *) fail "flash-attn 的 arch 和当前卡不匹配。重编: SKIP=0 bash 03_build_ext.sh" ;;
esac

echo ">>> 数据       : $TRAIN_FILE"
echo ">>> checkpoint : $CKPT_DIR"
if [ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
  echo ">>> ⚠️  发现已有 checkpoint (step $(cat "$CKPT_DIR/latest_checkpointed_iteration.txt"))，"
  echo "        resume_mode=auto 会从那里续训。想从头开始就先 rm -rf $CKPT_DIR"
fi
echo

# ---------------------------------------------------------------------------
# 开跑
# ---------------------------------------------------------------------------
# 09_train_smoke.sh 负责用 sed 从仓库的 scripts/run_vision_opd.sh 生成配置副本
# (仓库代码保持不动)。它自己会传 trainer.save_freq=-1 等默认值, 但因为末尾转发了
# "$@", 我们下面这些 override 排在 hydra 命令行更后面, **后者生效**。
#
# 相对论文原设置, 只有这些偏离, 且每一条都有理由:
#   TRAIN_BATCH_SIZE 96→24     资源限制(768 序列需要约 890 GiB driver 内存)
#   TRAINER_N_GPUS   8→2       资源限制
#   三个 offload    True→False  论文为 8×80GB 设计; 我们 2×96GB 不需要,
#                              且实测开着慢 3.5 倍。**不改变梯度**, 纯性能选择
#   ROLLOUT_GPU_MEM 0.7→0.45   只影响 vLLM KV cache 大小, 不改变采样结果
#   *_NUM_WORKERS   8→2        并发度, 纯吞吐
#   lr_warmup_steps 10→40      **batch 改动的衍生后果**: 论文 10/65 步 = 15%,
#                              按 warmup 期间看到的数据量对齐, 我们需要 40/260
# lr 保持 2e-6 不变 —— 一次只改一个变量(batch), 结果好坏才能干净归因。
# 其余全部保持论文原值: alpha=0.5 / distillation_topk=100 / is_clip=2.0 /
# teacher_regularization=ema / teacher_update_rate=0.05 / rollout_is=token 阈值 2.0 /
# clip_ratio / max_prompt_length=8192 / max_response_length=1024 /
# ppo_max_token_len_per_gpu=9216 / gradient_checkpointing / use_remove_padding
# experiment_name 也一并覆盖: 上游是 EXPERIMENT_NAME="Vision-OPD-${MODEL_NAME}"
# (run_vision_opd.sh:55), **与数据文件无关** —— 视觉特权版和语言特权版会算出同一个名字,
# 日志和 checkpoint 目录都分不开。
bash "$SETUP/09_train_smoke.sh" \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.experiment_name=vopd-bboxtext-2gpu \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.rollout_data_dir="$ROLLOUT_DIR" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=40
rc=$?

echo
echo "=========================================================="
echo " end   : $(date)   exit=$rc"
echo " ckpt  : $(ls -1 "$CKPT_DIR" 2>/dev/null | tr '\n' ' ')"
echo " 峰值内存(MaxRSS): $(sstat -j "${SLURM_JOB_ID:-0}" --format=MaxRSS -a -n 2>/dev/null | tr -d ' ' | head -1)"
echo "=========================================================="
exit $rc
