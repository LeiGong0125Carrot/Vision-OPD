#!/bin/bash
# 训练冒烟测试: 跑 1-2 步就停, 只为确认训练代码能跑起来, 不追求收敛。
#
# 目的是一次性暴露这几个至今完全没验证过的东西:
#   1) pip 装的 flash_attn 首次被真正执行 (推理走的是 vLLM 自带的 FA, 不是这个)
#   2) FSDP + Ray 起得来吗
#   3) 单卡塞得下 FSDP actor + ref model + vLLM rollout 吗
#   4) vopd 自蒸馏 loss / teacher forward / FSDP<->vLLM 权重同步
#   5) parquet 数据列名和训练配置对不对得上
#
# 做法: 不改仓库代码, 而是 sed 出一份 run_vision_opd.sh 的单卡小配置副本再跑。
# 这样上游脚本更新时我们只需要重新生成, 不会有改动冲突。
set -euo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh

SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
SRC="$VOPD_ROOT/scripts/run_vision_opd.sh"
DST="$SETUP/run_vision_opd_smoke.sh"
STEPS="${STEPS:-2}"

# 原脚本把 actor 参数/优化器状态/ref 模型全部 offload 到 CPU。那是给 8 卡设计的 ——
# 每卡显存紧张时往 CPU 挪是对的。但单卡 B200 的情况正好相反:
#     显存 183GB (实测只用了 53GB, 余 130GB)   vs   CPU 内存 128GB (Slurm --mem 上限)
# 开着 offload 跑, cgroup 会精确撞满 128GiB (memory.failcnt=70), OOM killer 杀掉
# WorkerDict, Ray 那边表现为 "ActorUnavailableError: Connection reset by peer"。
# 所以单卡默认关掉 offload, 让参数和优化器状态留在富余的显存里。
#   OFFLOAD=True bash 09_train_smoke.sh   # 想恢复 offload(比如 CPU 内存申请得够大)
OFFLOAD="${OFFLOAD:-False}"
# 三个开关可以分开控制(默认跟随 OFFLOAD)。实测数据支持"混合 offload":
#   显存峰值 171.68/183.36 GB (93.6%, 全关 offload 时) —— 显存才是瓶颈, 不是 CPU。
#   按访问频率分配最优:
#     actor 参数  ~24GB  每次前向/反向都用      -> 留 GPU  (PARAM_OFFLOAD=False)
#     AdamW 动量  ~32GB  每步只 optimizer_step 一次(实测 0.05s) -> 放 CPU (OPT_OFFLOAD=True)
#     ref 模型    ~8GB   每步一次 log_prob      -> 放 CPU  (REF_OFFLOAD=True)
#   这样显存能从 171GB 降到 ~120-130GB, CPU 侧只需约 50-80GB。
# 用法: OPT_OFFLOAD=True REF_OFFLOAD=True bash 09_train_smoke.sh
PARAM_OFFLOAD="${PARAM_OFFLOAD:-$OFFLOAD}"
OPT_OFFLOAD="${OPT_OFFLOAD:-$OFFLOAD}"
REF_OFFLOAD="${REF_OFFLOAD:-$OFFLOAD}"

# 显存/批量旋钮。不够时按此顺序往下调: ROLLOUT_GPU_MEM -> RN -> BATCH
BATCH="${BATCH:-4}"                          # train_batch_size (= ppo_mini_batch_size)
RN="${RN:-2}"                                # rollout.n
ROLLOUT_GPU_MEM="${ROLLOUT_GPU_MEM:-0.35}"   # vLLM 显存占比(训练阶段它 sleep 释放 KV cache)

# 卡数。默认按当前作业实际分到的 GPU 数, 拿不到就 1。
# 注意 >1 时 FSDP 从 NO_SHARD 变成 FULL_SHARD, 静态显存(参数/梯度/master/AdamW/teacher,
# 单卡实测约 81 GiB)会被 N 等分 —— 这正是 96GB 的 RTX PRO 6000 装得下的原因。
_ngpu_detect() {
  # `|| true` 不能省: grep -c 数到 0 时**退出码是 1**, 在本脚本的 set -e 下会让
  # 这个赋值失败 -> 整个脚本静默退出, 下面那行兜底根本执行不到。
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | grep -c "^GPU" || true
  else
    echo 1
  fi
}
NGPUS="${NGPUS:-$(_ngpu_detect || true)}"
[ "${NGPUS:-0}" -ge 1 ] 2>/dev/null || NGPUS=1

# 训练数据。默认是 08 生成的原版(视觉特权); 换成 12 生成的"语言特权"版:
#   TRAIN_FILE=$VOPD_ROOT/data/train_bbox_text.parquet bash 09_train_smoke.sh
TRAIN_FILE="${TRAIN_FILE:-$VOPD_ROOT/data/train.parquet}"

# --- 前置检查 ---
if [ ! -f "$TRAIN_FILE" ]; then
  echo "❌ 找不到训练数据: $TRAIN_FILE"
  echo "   原版数据:     bash 08_prep_train_data.sh"
  echo "   语言特权版:   python 12_prep_data_bbox_text.py"
  exit 1
fi
echo ">>> 训练数据: $TRAIN_FILE"

# teacher 走哪条通路, 完全由 parquet 里有没有 `teacher_prompt` 列决定
# (ray_trainer.py:1300)。没有这一列时 _prepare_teacher_messages(:801) 会**静默**
# 回退到 _swap_images_in_messages —— 不报错、不打日志、不记 metric。
#
# 在「语言特权」设定下这个回退是灾难性的: 因为 bbox_images 和 images 是同一张干净
# 原图, 回退后 teacher prompt 会和 student **逐字节相同**, 蒸馏退化成自己蒸自己,
# JSD≈0, 训练看起来一切正常但学不到任何东西。所以这里显式判定并打印出来。
"$VOPD_PY" - "$TRAIN_FILE" <<'PY'
import sys
import pyarrow.parquet as pq
cols = pq.read_schema(sys.argv[1]).names
need = [c for c in ("prompt", "images", "bbox_images") if c not in cols]
if need:
    sys.exit(f"❌ parquet 缺少必需列 {need}；实际列: {cols}")
if "teacher_prompt" in cols:
    print("    teacher 通路: teacher_prompt 模板 (语言特权 —— 文本里给 bbox 坐标)")
else:
    print("    teacher 通路: ⚠️  图像替换回退 (原版视觉特权 —— teacher 看 bbox_images)")
    print("       若你本意是跑语言特权版, 现在就停下: 忘了带")
    print("       TRAIN_FILE=$VOPD_ROOT/data/train_bbox_text.parquet")
PY
[ $? -eq 0 ] || exit 1

# vLLM server 占着 85% 显存, 训练要用整张卡
if [ -f "$SETUP/.serve.pid" ] && kill -0 "$(cat "$SETUP/.serve.pid")" 2>/dev/null; then
  echo ">>> 先停掉 vllm server（它占着 85% 显存）"
  bash "$SETUP/06_serve.sh" stop; sleep 5
fi
if [ -f "$SETUP/.keepalive.pid" ] && kill -0 "$(cat "$SETUP/.keepalive.pid")" 2>/dev/null; then
  echo ">>> 先停掉 keepalive"; bash "$SETUP/keepalive.sh" stop; sleep 3
fi
echo ">>> 当前显存: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"

# --- 生成单卡小配置副本 ---
echo ">>> 从 $SRC 生成冒烟配置"
# PROJECT_ROOT 必须一起改: 原脚本第 15 行是
#     PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# 它假定自己住在 <repo>/scripts/ 下, 所以 dirname/.. 正好是仓库根。而我们的副本放在
# Vision-OPD-setup/ 里, 那个 ".." 会跳到 /sfs/weka/scratch/nkw3mr, 导致
# chat_templates/ 和 data/ 的路径全错。直接写死成真正的仓库根。
sed -E \
  -e "s|^PROJECT_ROOT=.*|PROJECT_ROOT=\"$VOPD_ROOT\"|" \
  -e "s|^TASK_TRAIN_FILE=.*|TASK_TRAIN_FILE=\"$TRAIN_FILE\"|" \
  -e "s/^TRAIN_BATCH_SIZE=.*/TRAIN_BATCH_SIZE=$BATCH/" \
  -e "s/^PPO_MIMI_BATCH_SIZE=.*/PPO_MIMI_BATCH_SIZE=$BATCH/" \
  -e "s/^ROLLOUT_N=.*/ROLLOUT_N=$RN/" \
  -e "s/^ROLLOUT_GPU_MEMORY_UTILIZATION=.*/ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEM/" \
  -e "s/^TRAINER_N_GPUS_PER_NODE=.*/TRAINER_N_GPUS_PER_NODE=$NGPUS/" \
  -e 's/^TRAINER_NNODES=.*/TRAINER_NNODES=1/' \
  -e "s/^ACTOR_PARAM_OFFLOAD=.*/ACTOR_PARAM_OFFLOAD=$PARAM_OFFLOAD/" \
  -e "s/^ACTOR_OPTIMIZER_OFFLOAD=.*/ACTOR_OPTIMIZER_OFFLOAD=$OPT_OFFLOAD/" \
  -e "s/^REF_PARAM_OFFLOAD=.*/REF_PARAM_OFFLOAD=$REF_OFFLOAD/" \
  -e 's/^TRAINER_LOGGER=.*/TRAINER_LOGGER='"'"'["console"]'"'"'/' \
  -e 's/^ROLLOUT_AGENT_NUM_WORKERS=.*/ROLLOUT_AGENT_NUM_WORKERS=2/' \
  -e 's/^DATA_DATALOADER_NUM_WORKERS=.*/DATA_DATALOADER_NUM_WORKERS=2/' \
  "$SRC" > "$DST"
chmod +x "$DST"

# 这里**不用 diff**。diff 只显示有差异的行, 而一条 sed 规则失效时那行两边一模一样,
# 于是不出现在 diff 里 —— 和"这个值本来就没打算改"完全无法区分。(比如 RN=8 时
# ROLLOUT_N 和上游一字不差, diff 什么都不显示, 看不出规则到底命中没有。)
# 直接把生成结果里的实际值打出来, 无歧义; 并断言每条 sed 规则都真的产生了预期值。
echo ">>> 生成配置的实际取值:"
_check() {   # _check <变量名> <期望值>
  local got
  got=$(grep -m1 -E "^$1=" "$DST" | cut -d= -f2- | tr -d '"')
  printf "    %-32s %s\n" "$1" "$got"
  if [ -n "${2:-}" ] && [ "$got" != "$2" ]; then
    echo "    ❌ 期望 [$2] —— sed 规则没命中, 中止"; exit 1
  fi
}
_check PROJECT_ROOT                    "$VOPD_ROOT"
_check TASK_TRAIN_FILE                 "$TRAIN_FILE"
_check TRAIN_BATCH_SIZE                "$BATCH"
_check PPO_MIMI_BATCH_SIZE             "$BATCH"
_check ROLLOUT_N                       "$RN"
_check ROLLOUT_GPU_MEMORY_UTILIZATION  "$ROLLOUT_GPU_MEM"
_check TRAINER_N_GPUS_PER_NODE         "$NGPUS"
_check TRAINER_NNODES                  "1"
_check ACTOR_PARAM_OFFLOAD             "$PARAM_OFFLOAD"
_check ACTOR_OPTIMIZER_OFFLOAD         "$OPT_OFFLOAD"
_check REF_PARAM_OFFLOAD               "$REF_OFFLOAD"
_check PPO_MAX_TOKEN_LEN_PER_GPU       ""   # 仅展示(值是 $TRAIN_MAX_MODEL_LEN, 未被 sed)
_check LR                              ""

# --- 环境 ---
# 原脚本第 39 行是 TRAINER_NNODES=$WORLD_SIZE, 而它 set -u -> 未定义就直接 unbound variable。
# 我们已经 sed 成 1 了, 这里再 export 一份兜底(万一上游改了写法)。
export WORLD_SIZE=1
export RAY_TMPDIR="$TMPDIR/ray"          # ray 默认写 /tmp, 节点上 /tmp 可能很小
# 注意: RAY_DISABLE_MEMORY_MONITOR 只被 ray 的遗留 Python MemoryMonitor 读取,
# 而那个模块在运行时链路上没人 import —— 真正杀 WorkerDict 的是 raylet 里的 C++
# memory monitor, 开关是下面这个。之前那次 128GiB cgroup OOM, 旧的那行没起作用。
export RAY_memory_monitor_refresh_ms=0
mkdir -p "$RAY_TMPDIR"

LOG="$SETUP/train_smoke.log"
echo
echo ">>> 启动训练, 限制 $STEPS 步 (日志: $LOG)"
echo ">>> 基座模型 Qwen/Qwen3.5-4B 首次会下载约 8GB"
echo

cd "$VOPD_ROOT"
# EXTRA_ARGS 会被原脚本追加到 hydra 命令行末尾。
#
# 末尾的 "$@" 把本脚本收到的多余参数继续往下透传, 于是任何 hydra override 都能直接用,
# 不必为每个配置项都造一个环境变量旋钮。重复的 key **后者生效**。
#
# (先前这里写的论据是错的: 说"run_vision_opd.sh:156 已有 trainer.save_freq, 下面又传
#  一个, 两个并存还能跑"—— 但那两个值都是 -1, 只能证明 hydra 不报错, 证明不了后者
#  生效。真正的证据是用本项目配置实测 compose:
#      compose("vopd", overrides=["trainer.save_freq=-1", "trainer.save_freq=77"])
#      -> save_freq=77
#  以及真实日志里 ppo_max_token_len_per_gpu 出现两次时, 生效的是排在最后的那个。)
# 因为 "$@" 排在最后, 所以它能覆盖上面这几个默认值。例如全量跑:
#   bash 09_train_smoke.sh trainer.total_training_steps=260 \
#        trainer.save_freq=20 trainer.max_actor_ckpt_to_keep=2 \
#        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18432
bash "$DST" \
    trainer.total_training_steps="$STEPS" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    "$@" \
    2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo
if [ "$rc" -eq 0 ]; then
  echo "✅ 训练冒烟测试通过 —— 训练代码能跑"
  grep -E "step:|Total training steps|timing_s/" "$LOG" | tail -15
else
  echo "❌ 失败 (exit $rc)。关键报错:"
  grep -nE "Error|Traceback|CUDA out of memory|RuntimeError|AssertionError|unbound" "$LOG" | tail -20
fi
echo
# 注意: nvidia-smi 给的是**此刻**的占用, 不是峰值(训练已结束, vLLM 也睡了)。
# 真正的峰值在 metrics 里, 且单位是 GiB (fsdp_workers.py:981 用的是 /1024**3):
#   perf/max_memory_allocated_gb  真实张量峰值   <- 看这个
#   perf/max_memory_reserved_gb   保留的虚拟地址空间; 开了 expandable_segments 后
#                                 可以超过物理显存, 不代表真的用了那么多
echo ">>> 当前显存(非峰值): $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
echo ">>> 每步显存峰值 / CPU 内存:"
grep -oE "step:[0-9]+ .*" "$LOG" | tr ' ' '\n' \
  | grep -E "^(step:|perf/max_memory_allocated_gb:|perf/max_memory_reserved_gb:|perf/cpu_memory_used_gb:|timing_s/step:)" \
  | paste - - - - - 2>/dev/null | sed 's/^/    /' || true
exit $rc
