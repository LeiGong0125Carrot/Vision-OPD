#!/bin/bash
# 「坐标作为输出、特权放 system」版的本地冒烟测试 —— 跑 2 步就停。
#
#   bash 21_smoke_bboxout.sh          # 默认 2 步
#   STEPS=5 bash 21_smoke_bboxout.sh
#
# 和正式跑(20_train_bboxout.sbatch)的唯一差别是 total_training_steps / save_freq /
# 目录名。其余每个 hydra 参数逐字节相同 —— 冒烟测的就是正式跑会走的那条路径。
#
# 同样**直接调用仓库的 scripts/run_vision_opd.sh**，不做副本、不改仓库代码。
set -uo pipefail
SETUP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SETUP/00_env.sh"

STEPS="${STEPS:-2}"
TRAIN_FILE="${TRAIN_FILE:-$VOPD_ROOT/data/train_bbox_output.parquet}"
CKPT="$VOPD_ROOT/checkpoints/_smoke_bboxout"
ROLLOUT="$VOPD_ROOT/rollouts/_smoke_bboxout"
LOG="$SETUP/smoke_bboxout.log"

fail() { echo "❌ $*"; exit 1; }

# ---------------------------------------------------------------- 前置检查
[ -f "$TRAIN_FILE" ] || fail "缺训练数据: $TRAIN_FILE  (先跑 19_prep_data_bbox_output.py)"

n=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || true)
[ "${n:-0}" -ge 2 ] || fail "只看到 $n 张卡，需要 2 张"

# GPU 必须是空的。vLLM 服务(评测用的 8765/8766)会占掉几乎全部显存，而且
# 06_serve.sh stop 杀不掉子进程 VLLM::EngineCore —— 踩过，要 nvidia-smi 找 PID 后 kill -9。
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2>2000{print "  GPU "$1": "$2" MiB"}')
if [ -n "$busy" ]; then
  echo "❌ GPU 上还有东西占着显存:"; echo "$busy"
  echo "   先停掉评测服务:"
  echo "     PORT=8766 bash $SETUP/06_serve.sh stop"
  echo "     PORT=8765 bash $SETUP/06_serve.sh stop"
  echo "   若仍有残留(VLLM::EngineCore)，nvidia-smi 查 PID 后 kill -9"
  exit 1
fi

want=$(cat "$VOPD_ENV/.flash_attn_archs" 2>/dev/null)
have=$(cuobjdump --list-elf "$VOPD_ENV"/lib/python3.12/site-packages/flash_attn_2_cuda*.so 2>/dev/null \
       | grep -oE "sm_[0-9]+" | sort -u | tr '\n' ' ')
case "$have" in *"sm_$want"*) ;; *) fail "flash-attn 编译为 [$have]，和当前卡(期望 sm_$want)不匹配" ;; esac

# teacher 走哪条通路完全由 parquet 有没有 teacher_prompt 列决定(ray_trainer.py:1300)。
# 没有就**静默**回退到图像替换 —— 而我们两边是同一张图，等于自己蒸自己，JSD≈0 却不报错。
"$VOPD_PY" - "$TRAIN_FILE" <<'PY' || exit 1
import sys, pyarrow.parquet as pq
cols = pq.read_schema(sys.argv[1]).names
if "teacher_prompt" not in cols:
    sys.exit(f"❌ parquet 缺 teacher_prompt 列；实际: {cols}")
PY

# Ray 的 AF_UNIX socket 上限 107 字节，Ray 自己的后缀固定占 67 -> 这里最多 40。
# $TMPDIR/ray_<jobid> 是 41 字节，超 1 个字节就起不来（踩过）。
export RAY_TMPDIR="$TMPDIR/ray_bo"
[ ${#RAY_TMPDIR} -le 40 ] || fail "RAY_TMPDIR 过长(${#RAY_TMPDIR}>40)"
export RAY_memory_monitor_refresh_ms=0     # 真正杀 WorkerDict 的是 raylet 的 C++ monitor
export WORLD_SIZE=1                        # run_vision_opd.sh:41 在 set -u 下用它
mkdir -p "$RAY_TMPDIR" "$CKPT" "$ROLLOUT"  # 仓库脚本只 mkdir 它自己算的 rollout 路径

# resume_mode 默认是 auto，会自动续上旧 checkpoint —— 冒烟要从头开始
rm -rf "$CKPT"; mkdir -p "$CKPT"

echo "=========================================================="
echo " 冒烟: 坐标作为输出 + 特权放 system   ($STEPS 步)"
echo " GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ×$n"
echo " 数据 : $TRAIN_FILE"
echo " 日志 : $LOG"
echo "=========================================================="
echo

cd "$VOPD_ROOT"
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
    trainer.total_training_steps="$STEPS" \
    trainer.save_freq=-1 \
    trainer.experiment_name=smoke-bboxout \
    trainer.default_local_dir="$CKPT" \
    trainer.rollout_data_dir="$ROLLOUT" \
    2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

echo
echo "=========================================================="
if [ "$rc" -ne 0 ]; then
  echo "❌ 失败 (exit $rc)，关键报错:"
  grep -nE "Error|Traceback|CUDA out of memory|OutOfMemory|unbound|AF_UNIX" "$LOG" | tail -12 | cut -c1-190
  exit $rc
fi
echo "✅ 跑完了。下面三项才是这次要看的:"
echo
echo "--- ① teacher 通路（swap_fraction 必须是 1.0；JSD 不能≈0）---"
grep -oE "teacher_image_swap_fraction:[0-9.]+|self_distillation_mask.mean\(\):[0-9.]+|raw_jsd_token_mean:[0-9.e-]+|actor/grad_norm:[0-9.e-]+" "$LOG" | tail -8 | sed 's/^/    /'
echo
echo "--- ② 泄漏检查（上一版 191/191 都是「Based on the analysis of...」+ 捏造坐标）---"
f=$(ls -t "$ROLLOUT"/*.jsonl 2>/dev/null | head -1)
if [ -n "$f" ]; then
  "$VOPD_PY" - "$f" <<'PY'
import json, sys, collections, re
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"    n={len(rows)}  回复长度中位={sorted(len(str(r['output'])) for r in rows)[len(rows)//2]}")
op=collections.Counter(str(r['output'])[:40] for r in rows)
print(f"    开头最常见的 3 种:")
for k,v in op.most_common(3): print(f"      [{v:3}] {k!r}")
nb=sum(1 for r in rows if re.search(r'"bbox_2d"', str(r['output'])))
print(f"    含 bbox_2d 的: {nb}/{len(rows)}")
print(f"    含 'Based on the analysis': {sum(1 for r in rows if 'Based on the analysis' in str(r['output']))}/{len(rows)}  ← 应为 0")
print(f"\n    前 2 条完整输出:")
for r in rows[:2]: print(f"      {str(r['output'])[:200]!r}")
PY
else
  echo "    (没找到 rollout dump: $ROLLOUT)"
fi
echo
echo "--- ③ 显存 / 步时 ---"
grep -oE "step:[0-9]+|max_memory_allocated_gb:[0-9.]+|cpu_memory_used_gb:[0-9.]+|timing_s/step:[0-9.]+" "$LOG" | paste - - - - 2>/dev/null | tail -5 | sed 's/^/    /'
echo "=========================================================="
