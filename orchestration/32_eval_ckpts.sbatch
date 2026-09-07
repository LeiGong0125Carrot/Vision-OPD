#!/bin/bash
# 把 vopd-xy_rb-e1 的 checkpoint 在 V* 上测一遍, 顺带用同一条流水线重测 base。
#
# 为什么要重测 base:
#   eval/judge/vstar/ 里已有 base-Qwen3.5-4B 81.15% / base-nothink 80.63% /
#   vopd-b24-step260 72.77%, 但那几次用了 LLM judge(43~52 条走了 LLM), 而且
#   当时的 chat template 没有记录。这里全部用 NO_JUDGE=1(纯规则) + 训练同款
#   template 重跑, 四个数才是同一把尺子量出来的。
#   NO_JUDGE 对 MCQ 是**严格下界**(judge_qwenlm.py:247-270 里 LLM 只在规则判错时
#   才被调用, 只能把错翻成对), 所以横向比较有效, 但绝对值会低于带 judge 的数。
#
# 为什么测 130/195/260 三个:
#   训练集上 IoU 在 step ~170 见顶后崩到 0.095, 但崩的是输出格式不是定位能力
#   (能解析出的框 IoU 末期仍有 0.73)。而 V* 上模型输出的是裸字母、不受 JSON
#   格式漂移影响 —— 所以 V* 能干净地回答"推理能力到底是被训好了还是训坏了"。
#   上一轮 b24 的 step260 掉到 72.77%(base 81.15%), 如果那是"训过头", 130 应该
#   明显好于 260; 如果 130 也掉, 那就是这个训练目标本身在伤害答题能力。
#
# 卡型: B200 (sm_100) —— **评测可以换卡, 训练不行**, 区别在这:
#   pip 装的 flash_attn_2_cuda.so 里只有 sm_120 cubin 且不带 PTX
#   (03_build_ext.sh 单 arch 编译; 2026-08-26 cuobjdump 实测确认)。
#   但 vllm/v1/attention/backends/fa_utils.py:19-23 里 CUDA 分支走的是
#   `from vllm.vllm_flash_attn import ...` —— vllm **自带**的 FA, 外部
#   flash_attn 包只在 ROCm 分支才 import。所以:
#     训练(verl 直接调 flash_attn) -> 锁死 sm_120, 只能 rtxpro6000
#     推理(vllm 自带 FA + FlashInfer JIT) -> arch 无关, B200 可跑
#   佐证: 整个 env 本来就是在 B200 上装的(STATUS.md 01/02 行), Vision-OPD-9B
#   那次 90.05% 也是 B200 测的。
#   万一视觉塔的注意力后端选歪了, 安全阀是 vllm 的 --mm-encoder-attn-backend TORCH_SDPA
#   (vllm/config/multimodal.py:156)。
#SBATCH --job-name=vopd-eval
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/eval-%j.log
#SBATCH --open-mode=append

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
mkdir -p "$SETUP/logs"

TAG="${TAG:-vopd-xy_rb-e1}"
CKPT_ROOT="$VOPD_ROOT/checkpoints/$TAG"
MERGE_ROOT="$VOPD_ROOT/merged"
STEPS="${STEPS:-130 195 260}"
# 端口避开 8000/8765/8766 —— GPU 节点是共享的, 别人也在起 vLLM。
export PORT="${PORT:-8811}"
BENCH="${BENCH:-vstar}"

echo "=========================================================="
echo " V* 评测   tag=$TAG   steps=$STEPS   bench=$BENCH"
echo " 节点=$(hostname)  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo " 判分: NO_JUDGE=1 (纯规则, 严格下界)"
echo " $(date)"
echo "=========================================================="

# ---------- 1. 合并 FSDP 分片 ----------
# model_merger 写到独立目录, 不碰 checkpoints/ 里的原始分片
# (scripts/merge_checkpoint.sh 是就地合并, 会 delete 掉顶层文件, 这里不用它)。
for s in $STEPS; do
  SRC="$CKPT_ROOT/global_step_$s/actor"
  DST="$MERGE_ROOT/$TAG-step$s"
  if [ -f "$DST/config.json" ] && ls "$DST"/*.safetensors >/dev/null 2>&1; then
    echo ">>> step $s 已合并, 跳过 ($(du -sh "$DST" | cut -f1))"
    continue
  fi
  [ -d "$SRC" ] || { echo "❌ 缺 $SRC"; exit 1; }
  echo ">>> 合并 step $s: $SRC -> $DST"
  mkdir -p "$DST"
  ( cd "$VOPD_ROOT" && "$VOPD_PY" -m verl.model_merger merge \
      --backend fsdp --local_dir "$SRC" --target_dir "$DST" )
  rc=$?
  [ $rc -ne 0 ] && { echo "❌ step $s 合并失败 rc=$rc"; exit 1; }
  echo "    ✅ $(du -sh "$DST" | cut -f1)"
done

# ---------- 2. 逐个起服务 + 评测 ----------
run_one() {
  local name="$1" model="$2"
  echo
  echo "##########################################################"
  echo "# $name   <-  $model"
  echo "##########################################################"
  bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1
  sleep 5
  # CHAT_TEMPLATE 用 06_serve.sh 的默认值(perception_chat_template_qwen35.jinja),
  # 它预填空的 <think></think> —— 和训练时一致。不带的话模型会进思考模式,
  # 那是另一种行为, 跟训练分布对不上。
  MODEL="$model" SERVED_NAME="$name" bash "$SETUP/06_serve.sh" start
  if [ $? -ne 0 ]; then
    echo "❌ $name 起服务失败, 跳过"
    return 1
  fi
  OPENAI_MODEL_ID="$name" NO_JUDGE=1 PARALLEL_WORKERS=32 \
    bash "$SETUP/07_eval.sh" "$BENCH"
  local rc=$?
  bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1
  sleep 5
  return $rc
}

# base 先跑 —— 它最快, 顺带验证整条流水线通。
run_one "base-qwen35-4b-rerun" "Qwen/Qwen3.5-4B"

for s in $STEPS; do
  run_one "e1-step$s" "$MERGE_ROOT/$TAG-step$s"
done

# ---------- 3. 汇总 ----------
echo
echo "=========================================================="
echo " 汇总 (纯规则判分, 严格下界)"
echo "=========================================================="
# 判分一律走 16_score_norule.py —— 它直接 import judge_qwenlm.py 的规则,
# 和仓库口径完全一致。这里连同旧的 base-* / vopd-b24-step260 一起重算一遍,
# 让新旧模型在**同一把尺子**下可比(旧的 judge/ 结果带 LLM judge, 不能直接混着比)。
for f in "$VOPD_ROOT/eval/model_answer/$BENCH"/*_answer.jsonl; do
  [ -e "$f" ] || continue
  m=$(basename "$f" _answer.jsonl)
  echo
  echo "--- $m ---"
  "$VOPD_PY" "$SETUP/16_score_norule.py" --benchmark "$BENCH" --model "$m" --by-category 2>&1 \
    | sed 's/^/    /'
done

echo
echo " 结束 $(date)"
