#!/bin/bash
# EVT 全流水线: 训练 -> 合并全部 checkpoint -> TreeBench(本地HF, 双卡并行)
#              -> V*(serve+推理) -> gpt-oss-120b 官方判卷(两基准) -> 汇总表。
#   sbatch 56_evt_pipeline.sbatch
#   变体: sbatch --export=ALL,EVT_ENABLE=False 跑 SA-v2 流水线 (目录/名字自动切换)
#
#SBATCH --job-name=evt-pipe
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=1000G
#SBATCH --time=2-00:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/pipeline-%j.log
#SBATCH --open-mode=append
set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"

export EVT_ENABLE="${EVT_ENABLE:-True}"
export EVT_NEGATIVE_ONLY="${EVT_NEGATIVE_ONLY:-False}"
export SA_ENABLE="${SA_ENABLE:-True}"
export EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX:-}"
# wandb 默认开启 (JSON 含逗号, 不能经 sbatch --export 传, 只能在此设定); 计算节点断网时自动退化为离线由 WANDB_MODE 控制
export TRAINER_LOGGER='["console","tensorboard","wandb"]'
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-5}"
export TRAINER_N_GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-2}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"

if [[ "$EVT_ENABLE" == "True" ]]; then
  if [[ "$EVT_NEGATIVE_ONLY" == "True" ]]; then
    TAG="EVTneg-Qwen3.5-4B"; SHORT="evtneg"
  else
    TAG="EVT-Qwen3.5-4B"; SHORT="evt4b"
  fi
elif [[ "$SA_ENABLE" == "True" ]]; then
  TAG="SA-OPDv2-Qwen3.5-4B"; SHORT="sav2"
else
  TAG="StdOPD-region-Qwen3.5-4B"; SHORT="v04b"   # 名字历史遗留, 实际形态由数据决定
fi
if [[ -n "$EXPERIMENT_SUFFIX" ]]; then
  TAG="${TAG}-${EXPERIMENT_SUFFIX}"; SHORT="${SHORT}-${EXPERIMENT_SUFFIX}"
fi
CKPT_ROOT="$VOPD_ROOT/checkpoints/$TAG"
MERGE_ROOT="$VOPD_ROOT/merged"
export PORT=8811   # 06_serve.sh / 07_eval.sh 都从环境读 PORT, 必须 export
JPORT=8813
JUDGE_MODEL=openai/gpt-oss-120b
VLLM_BIN="$(dirname "$VOPD_PY")/vllm"

echo "=========================================================="
echo " EVT 流水线  job=${SLURM_JOB_ID:-NA}  节点=$(hostname)  $(date)"
echo " 臂=$TAG  卡数=$TRAINER_N_GPUS_PER_NODE"
echo "=========================================================="

# ---------- 1. 训练 (完成→跳过; 半截→自动断点续训; 全新→从头训) ----------
# 预期总步数 = ceil(数据行数 / batch) × 1 epoch (与 run 脚本默认值保持镜像)
_TF="${TASK_TRAIN_FILE:-$VOPD_ROOT/data/TreeVGR-RL-37K/train_sa4k_hide.parquet}"
_TB="${TRAIN_BATCH_SIZE:-96}"
EXPECTED_STEPS=$("$VOPD_PY" -c "
import pyarrow.parquet as pq
print(pq.read_metadata('$_TF').num_rows // $_TB)" 2>/dev/null || echo 0)  # drop_last, 与 trainer 一致
LATEST=$(cat "$CKPT_ROOT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
if [ "$EXPECTED_STEPS" -gt 0 ] && [ "$LATEST" -ge "$EXPECTED_STEPS" ]; then
  echo ">>> 训练已完成 (step $LATEST >= 预期 $EXPECTED_STEPS), 跳过训练"
else
  echo ">>> 训练启动 (当前 step $LATEST / 预期 $EXPECTED_STEPS; verl resume_mode=auto 自动续训)"
  cd "$VOPD_ROOT"
  bash scripts/run_state_adaptive_4b.sh
  rc=$?
  [ $rc -ne 0 ] && { echo "❌ 训练失败 rc=$rc, 流水线终止"; exit $rc; }
fi

STEPS=$(ls -d "$CKPT_ROOT"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tr '\n' ' ')
echo ">>> checkpoints: $STEPS"
[ -z "$STEPS" ] && { echo "❌ 无 checkpoint"; exit 1; }

# ---------- 2. 合并 ----------
for s in $STEPS; do
  SRC="$CKPT_ROOT/global_step_$s/actor"; DST="$MERGE_ROOT/$TAG-step$s"
  if [ -f "$DST/config.json" ] && ls "$DST"/*.safetensors >/dev/null 2>&1; then
    echo ">>> step $s 已合并, 跳过"; continue
  fi
  echo ">>> 合并 step $s"; mkdir -p "$DST"
  ( cd "$VOPD_ROOT" && "$VOPD_PY" -m verl.model_merger merge \
      --backend fsdp --local_dir "$SRC" --target_dir "$DST" ) \
    || { echo "❌ step $s 合并失败"; exit 1; }
done

# ---------- 3. TreeBench 本地推理 (双卡并行, 每次两 ckpt) ----------
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
    echo ">>> vstar $name 答案已存在, 跳过 (防重复写入)"; continue
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
  > "$SETUP/logs/gptoss-pipe-${SLURM_JOB_ID:-x}.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
ok=0
for i in $(seq 1 150); do
  sleep 10
  kill -0 $SRV 2>/dev/null || { echo "❌ judge 服务退出"; tail -20 "$SETUP/logs/gptoss-pipe-${SLURM_JOB_ID:-x}.log"; exit 1; }
  curl -sf "http://localhost:$JPORT/v1/models" >/dev/null 2>&1 && { ok=1; break; }
done
[ $ok -eq 1 ] || { echo "❌ judge 25min 未就绪"; exit 1; }
echo "✅ judge 就绪"

# V* 判卷
cd "$VOPD_ROOT/eval"
VTAGS=""
for s in $STEPS; do
  tag="$SHORT-step${s}_seed42"
  [ -f "model_answer/vstar/${tag}_answer.jsonl" ] || continue
  VTAGS="$VTAGS $tag"
  "$VOPD_PY" judge_qwenlm.py --benchmark vstar --model "$tag" \
    --api_base "http://localhost:$JPORT/v1/" --judge_model "$JUDGE_MODEL"
done
# TreeBench 判卷 (merged 目录名小写 = infer_privilege 的 tag)
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
echo; echo "========== 最终汇总 (judge 口径; 参照: base TB 46.91 / V* 83.25; V0峰 TB 48.40 / V* 83.77) =========="
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
