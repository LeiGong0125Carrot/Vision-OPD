#!/bin/bash
# OPSA 评测流水线 (56_evt_pipeline 的 2-6 段, 适配 Vision-OPD-OPSA clone):
#   合并全部 ckpt (clone verl) -> TreeBench(双卡并行, 主仓库 harness)
#   -> V*(serve+推理) -> gpt-oss-120b 判卷(两基准) -> 汇总表。
# 输出全部落在主仓库 eval/ 标准位置, 与 base/EVT/V0 历史结果同表可比。
#   sbatch opsa/eval_opsa.sbatch
#   干跑臂: sbatch --export=ALL,EXPERIMENT_SUFFIX=dryrun opsa/eval_opsa.sbatch
#SBATCH --job-name=opsa-eval
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/opsa-eval-%j.log
#SBATCH --open-mode=append
set -uo pipefail
CLONE=/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"

# 臂可通过 env 覆盖: sbatch --export=ALL,ARM_TAG=Aha-Qwen3.5-4B,ARM_SHORT=aha4b ...
TAG="${ARM_TAG:-OPSA-Qwen3.5-4B}"; SHORT="${ARM_SHORT:-opsa4b}"
if [ -n "${EXPERIMENT_SUFFIX:-}" ]; then
  TAG="${TAG}-${EXPERIMENT_SUFFIX}"; SHORT="${SHORT}-${EXPERIMENT_SUFFIX}"
fi
CKPT_ROOT="$CLONE/checkpoints/$TAG"
MERGE_ROOT="$CLONE/merged"
export PORT=8811
JPORT=8813
JUDGE_MODEL=openai/gpt-oss-120b
VLLM_BIN="$(dirname "$VOPD_PY")/vllm"

echo "=========================================================="
echo " OPSA 评测  job=${SLURM_JOB_ID:-NA}  节点=$(hostname)  $(date)"
echo " 臂=$TAG  ckpt=$CKPT_ROOT"
echo "=========================================================="

STEPS=$(ls -d "$CKPT_ROOT"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tr '\n' ' ')
echo ">>> checkpoints: $STEPS"
[ -z "$STEPS" ] && { echo "❌ 无 checkpoint"; exit 1; }

# ---------- 2. 合并 (clone verl) ----------
for s in $STEPS; do
  SRC="$CKPT_ROOT/global_step_$s/actor"; DST="$MERGE_ROOT/$TAG-step$s"
  if [ -f "$DST/config.json" ] && ls "$DST"/*.safetensors >/dev/null 2>&1; then
    echo ">>> step $s 已合并, 跳过"; continue
  fi
  echo ">>> 合并 step $s"; mkdir -p "$DST"
  ( cd "$CLONE" && PYTHONPATH="$CLONE" "$VOPD_PY" -m verl.model_merger merge \
      --backend fsdp --local_dir "$SRC" --target_dir "$DST" ) \
    || { echo "❌ step $s 合并失败"; exit 1; }
done

# ---------- 3. TreeBench 本地推理 (双卡并行, 主仓库 harness) ----------
cd "$VOPD_ROOT/eval/treebench_probe"
arr=($STEPS)
for ((i=0; i<${#arr[@]}; i+=2)); do
  s0=${arr[i]}
  CUDA_VISIBLE_DEVICES=0 "$VOPD_PY" infer_privilege.py \
    --model "$MERGE_ROOT/$TAG-step$s0" --privilege none --no-think &
  P0=$!
  P1=""
  if [ $((i+1)) -lt ${#arr[@]} ]; then
    s1=${arr[i+1]}
    CUDA_VISIBLE_DEVICES=1 "$VOPD_PY" infer_privilege.py \
      --model "$MERGE_ROOT/$TAG-step$s1" --privilege none --no-think &
    P1=$!
  fi
  wait $P0 || echo "⚠️ treebench step$s0 推理非零退出"
  [ -n "$P1" ] && { wait $P1 || echo "⚠️ treebench step${arr[i+1]} 推理非零退出"; }
done
echo ">>> TreeBench 推理完成 $(date)"

# ---------- 4. V* serve + 推理 (GPU0 顺序) ----------
for s in $STEPS; do
  name="$SHORT-step$s"
  if [ -f "$VOPD_ROOT/eval/model_answer/vstar/${name}_seed42_answer.jsonl" ]; then
    echo ">>> vstar $name 答案已存在, 跳过"; continue
  fi
  bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1; sleep 5
  CUDA_VISIBLE_DEVICES=0 MODEL="$MERGE_ROOT/$TAG-step$s" SERVED_NAME="$name" \
    bash "$SETUP/06_serve.sh" start || { echo "⚠️ $name 起服务失败, 跳过"; continue; }
  OPENAI_MODEL_ID="$name" NO_JUDGE=1 PARALLEL_WORKERS=32 \
    bash "$SETUP/07_eval.sh" vstar || echo "(NO_JUDGE 判分步预期非零)"
  bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1; sleep 5
done
echo ">>> V* 推理完成 $(date)"

# ---------- 5. gpt-oss 判卷 (两基准) ----------
"$VLLM_BIN" serve "$JUDGE_MODEL" --port "$JPORT" \
  --gpu-memory-utilization 0.92 --max-model-len 8192 \
  > "$SETUP/logs/gptoss-opsa-${SLURM_JOB_ID:-x}.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
ok=0
for i in $(seq 1 150); do
  sleep 10
  kill -0 $SRV 2>/dev/null || { echo "❌ judge 服务退出"; tail -20 "$SETUP/logs/gptoss-opsa-${SLURM_JOB_ID:-x}.log"; exit 1; }
  curl -sf "http://localhost:$JPORT/v1/models" >/dev/null 2>&1 && { ok=1; break; }
done
[ $ok -eq 1 ] || { echo "❌ judge 25min 未就绪"; exit 1; }
echo "✅ judge 就绪"

cd "$VOPD_ROOT/eval"
for s in $STEPS; do
  tag="$SHORT-step${s}_seed42"
  [ -f "model_answer/vstar/${tag}_answer.jsonl" ] || continue
  "$VOPD_PY" judge_qwenlm.py --benchmark vstar --model "$tag" \
    --api_base "http://localhost:$JPORT/v1/" --judge_model "$JUDGE_MODEL"
done
cd "$VOPD_ROOT/eval/treebench_probe"
TBM=""
for s in $STEPS; do
  low=$(echo "$TAG-step$s" | tr 'A-Z' 'a-z')
  [ -f "$VOPD_ROOT/eval/model_answer/treebench/${low}-priv-none-nothink_answer.jsonl" ] && \
    TBM="$TBM ${low}-priv-none-nothink"
done
[ -n "$TBM" ] && "$VOPD_PY" judge_treebench.py \
  --api-base "http://localhost:$JPORT/v1/" --judge-model "$JUDGE_MODEL" --models $TBM
kill $SRV 2>/dev/null; trap - EXIT

# ---------- 6. 汇总 ----------
echo; echo "========== 最终汇总 (judge 口径; 参照: base TB 46.91 / V* 83.25; V0峰 48.40/83.77; 同数据 V0-6karmA TB峰 49.14@35) =========="
cd "$VOPD_ROOT/eval"
"$VOPD_PY" - <<PY
import json, os, re
steps = "$STEPS".split()
short = "$SHORT"; tag = "$TAG".lower()
print(f"{'step':>5} {'TreeBench(judge)':>17} {'V*(judge)':>10}")
for s in steps:
    tb = f"judge/treebench/{tag}-step{s}-priv-none-nothink_answer.jsonl"
    vs = f"judge/vstar/{short}-step{s}_seed42_answer.jsonl"
    tbv = vsv = "-"
    if os.path.exists(tb):
        recs = [json.loads(l) for l in open(tb)]
        y = sum(1 for r in recs if re.match(r"^yes\b", str(r.get("judge","")).strip(), re.I))
        tbv = f"{100*y/len(recs):.2f}%"
    if os.path.exists(vs):
        data = json.load(open(vs))
        y = sum(1 for it in data if re.match(r"^yes\b", str(it.get("judge","")).strip(), re.I))
        vsv = f"{100*y/len(data):.2f}%"
    print(f"{s:>5} {tbv:>17} {vsv:>10}")
PY
echo " 结束 $(date)"
