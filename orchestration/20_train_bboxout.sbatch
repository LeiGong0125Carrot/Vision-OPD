#!/bin/bash
# 「坐标作为输出、特权放 system」版训练。
#
#   sbatch 20_train_bboxout.sbatch
#
# **直接调用仓库自己的 scripts/run_vision_opd.sh**，不做任何副本、不改仓库代码。
# 我们要改的每一项都是 hydra override，追加在命令行末尾即可（run_vision_opd.sh:164
# 的 EXTRA_ARGS 在最后，重复 key 后者生效 —— 已用真实 Hydra compose 验证）。
#
# 之前那套 sed 生成副本的做法(09_train_smoke.sh)是多余的，而且引入了一堆自己的 bug：
# PROJECT_ROOT 要跟着改、并发时 DST/LOG/RAY_TMPDIR 撞车、sed 规则可能静默失效。
# 原地调用时 PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)" 本来就解析正确。
#
#SBATCH --job-name=vopd-bboxout
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/vopd-bboxout-%j.log
#SBATCH --open-mode=append

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
mkdir -p "$SETUP/logs"
source "$SETUP/00_env.sh"

echo "=========================================================="
echo " job   : ${SLURM_JOB_ID:-NA} on $(hostname)   $(date)"
echo " 实验  : 坐标作为输出成分 + 特权放 system"
echo "=========================================================="

TRAIN_FILE="$VOPD_ROOT/data/train_bbox_output.parquet"
CKPT="$VOPD_ROOT/checkpoints/vopd-bboxout-2gpu"
ROLLOUT="$VOPD_ROOT/rollouts/vopd-bboxout-2gpu"
# run_vision_opd.sh:59 只对它自己算出的 rollout 目录做 mkdir；我们用 hydra 覆盖了，
# 所以要自己建。
mkdir -p "$CKPT" "$ROLLOUT"

# --- 前置检查 ---
fail() { echo "❌ $*"; exit 1; }
[ -f "$TRAIN_FILE" ] || fail "缺训练数据: $TRAIN_FILE (先跑 19_prep_data_bbox_output.py)"
n=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || true)
[ "$n" = "2" ] || fail "拿到 $n 张卡，配置按 2 张调的"
want=$(cat "$VOPD_ENV/.flash_attn_archs" 2>/dev/null)
have=$(cuobjdump --list-elf "$VOPD_ENV"/lib/python3.12/site-packages/flash_attn_2_cuda*.so 2>/dev/null \
       | grep -oE "sm_[0-9]+" | sort -u | tr '\n' ' ')
echo ">>> GPU        : $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
echo ">>> flash-attn : [$have]，期望含 sm_$want"
case "$have" in *"sm_$want"*) ;; *) fail "flash-attn arch 不匹配当前卡" ;; esac

# teacher 走哪条通路完全由 parquet 有没有 teacher_prompt 列决定(ray_trainer.py:1300)；
# 没有就**静默**回退到图像替换，而我们两边是同一张图 -> 变成自己蒸自己，JSD≈0 却不报错。
"$VOPD_PY" - "$TRAIN_FILE" <<'PY' || exit 1
import sys, pyarrow.parquet as pq
cols = pq.read_schema(sys.argv[1]).names
if "teacher_prompt" not in cols:
    sys.exit(f"❌ parquet 缺 teacher_prompt 列；实际: {cols}")
print("    teacher 通路: teacher_prompt 模板 ✓")
PY

[ -f "$CKPT/latest_checkpointed_iteration.txt" ] && \
  echo ">>> ⚠️  已有 checkpoint (step $(cat "$CKPT/latest_checkpointed_iteration.txt"))，resume_mode=auto 会续训"

# Ray 的 AF_UNIX socket 路径上限 107 字节，Ray 自己的后缀固定占 67 -> 这里最多 40。
# $TMPDIR/ray_<jobid> 是 41 字节，超 1 个字节就起不来（踩过）。用定长短名。
export RAY_TMPDIR="$TMPDIR/ray_bo"
[ ${#RAY_TMPDIR} -le 40 ] || fail "RAY_TMPDIR 过长(${#RAY_TMPDIR}>40): $RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR"
# 真正杀 WorkerDict 的是 raylet 的 C++ memory monitor（RAY_DISABLE_MEMORY_MONITOR 是空操作）
export RAY_memory_monitor_refresh_ms=0
# run_vision_opd.sh:41 是 TRAINER_NNODES=$WORLD_SIZE，而它 set -u
export WORLD_SIZE=1

echo ">>> 数据: $TRAIN_FILE"
echo ">>> ckpt: $CKPT"
echo ">>> RAY_TMPDIR: $RAY_TMPDIR (${#RAY_TMPDIR}/40)"
echo

cd "$VOPD_ROOT"
# 以下每一项都是覆盖 run_vision_opd.sh 里的默认值。相对论文原设置的偏离及理由：
#   train_batch_size 96→24 / n_gpus 8→2      资源限制（768 序列需约 890GiB driver 内存）
#   三个 offload True→False                   论文为 8×80GB 设计；2×96GB 不需要，
#                                             实测开着慢 3.5 倍。**不改变梯度**
#   gpu_memory_utilization 0.7→0.45           只影响 vLLM KV cache，不改变采样结果
#   *_num_workers 8→2                         并发度，纯吞吐
#   lr_warmup_steps 10→40                     按 warmup 期间看到的数据量对齐论文
#                                             (论文 10×768 = 我们 40×192)
#   lr 保持 2e-6 不变 —— 一次只改一个变量
# 其余(alpha=0.5 / distillation_topk=100 / is_clip=2.0 / teacher_regularization=ema /
# teacher_update_rate=0.05 / rollout_is=token / clip_ratio / max_prompt_length /
# ppo_max_token_len_per_gpu / gradient_checkpointing)全部保持论文原值。
bash scripts/run_vision_opd.sh \
    data.train_files="[\"$TRAIN_FILE\"]" \
    data.train_batch_size=24 \
    data.dataloader_num_workers=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps=40 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.logger='["console"]' \
    trainer.total_training_steps=260 \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.experiment_name=vopd-bboxout-2gpu \
    trainer.default_local_dir="$CKPT" \
    trainer.rollout_data_dir="$ROLLOUT"
rc=$?

echo
echo "=========================================================="
echo " end : $(date)   exit=$rc"
echo " ckpt: $(ls -1 "$CKPT" 2>/dev/null | tr '\n' ' ')"
echo " MaxRSS: $(sstat -j "${SLURM_JOB_ID:-0}" --format=MaxRSS -a -n 2>/dev/null | tr -d ' ' | head -1)"
echo "=========================================================="
# 这次最该看的不是准确率，而是 rollout 里有没有再出现「捏造坐标 + 固定开场白」：
echo " 检查泄漏(前 3 条 rollout):"
head -3 "$ROLLOUT"/$(ls -t "$ROLLOUT" 2>/dev/null | head -1) 2>/dev/null | cut -c1-220
exit $rc
