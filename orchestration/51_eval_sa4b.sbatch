#!/bin/bash
# SA-OPD 4B pilot 双臂评测: 两臂 x {step 14,28,41} x {treebench, vstar}。
# 适配自 32_eval_ckpts.sbatch, 三点不同:
#   1. 双臂循环 (TAGS), 模型名前缀 sa4b-/v04b- 区分, 不再硬编码 e1-;
#   2. 每个模型起一次 vLLM 服务, 连测两个 bench 再停 (省 6 次服务启动);
#   3. 不重跑 base —— treebench 的 qwen3.5-4b-direct-nothink / vstar 的
#      base-qwen35-4b-rerun 都是同款 template + NO_JUDGE 口径, 直接可比。
# 判分: NO_JUDGE=1 纯规则 = 严格下界, 横向比较有效; 论文口径数字之后对
#       优胜 checkpoint 单独补官方 judge (treebench 上 judge 差 ~4 点)。
#
#   sbatch 51_eval_sa4b.sbatch
#
#SBATCH --job-name=sa4b-eval
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/eval-%j.log
#SBATCH --open-mode=append

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
mkdir -p "$SETUP/logs"

# ── 双队列赛跑守卫: 同一评测同时投了 b200 和 rtxpro6000 两个队列时,
#    谁先 RUNNING 谁干活; 后启动的发现对方在跑就立刻退出, 并顺手取消
#    对方仍在排队的副本。绝不允许两份同时写 model_answer/(draw 双跑教训)。
if [ -n "${OTHER_JOB:-}" ]; then
  other_state=$(squeue -j "$OTHER_JOB" -h -o %T 2>/dev/null | head -1)
  if [ "$other_state" == "RUNNING" ]; then
    echo "另一份评测 (job $OTHER_JOB) 已在运行, 本作业退出。"
    exit 0
  fi
  if [ -z "$other_state" ]; then
    st=$(sacct -j "$OTHER_JOB" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
    if [[ "$st" == COMPLETED* ]]; then
      echo "另一份评测 (job $OTHER_JOB) 已完成, 本作业退出。"
      exit 0
    fi
    echo "另一份评测 job $OTHER_JOB 已离队但状态=$st, 本作业接手重跑。"
  else
    echo "取消仍在排队的另一份评测 job $OTHER_JOB ($other_state)"
    scancel "$OTHER_JOB" 2>/dev/null
  fi
fi

TAGS="${TAGS:-SA-OPD-Qwen3.5-4B StdOPD-region-Qwen3.5-4B}"
STEPS="${STEPS:-14 28 41}"
BENCHES="${BENCHES:-treebench vstar}"
MERGE_ROOT="$VOPD_ROOT/merged"
export PORT="${PORT:-8811}"

short_name() {  # SA-OPD-Qwen3.5-4B -> sa4b ; StdOPD-region-Qwen3.5-4B -> v04b
  case "$1" in
    SA-OPD-*) echo sa4b ;;
    StdOPD-region-*) echo v04b ;;
    *) echo "$(echo "$1" | tr 'A-Z' 'a-z' | tr -cd 'a-z0-9' | head -c 8)" ;;
  esac
}

echo "=========================================================="
echo " SA-OPD 双臂评测  tags=$TAGS  steps=$STEPS  benches=$BENCHES"
echo " 节点=$(hostname)  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo " $(date)"
echo "=========================================================="

# ---------- 1. 合并 FSDP 分片 ----------
for tag in $TAGS; do
  for s in $STEPS; do
    SRC="$VOPD_ROOT/checkpoints/$tag/global_step_$s/actor"
    DST="$MERGE_ROOT/$tag-step$s"
    if [ -f "$DST/config.json" ] && ls "$DST"/*.safetensors >/dev/null 2>&1; then
      echo ">>> $tag step $s 已合并, 跳过"; continue
    fi
    [ -d "$SRC" ] || { echo "❌ 缺 $SRC"; exit 1; }
    echo ">>> 合并 $tag step $s"
    mkdir -p "$DST"
    ( cd "$VOPD_ROOT" && "$VOPD_PY" -m verl.model_merger merge \
        --backend fsdp --local_dir "$SRC" --target_dir "$DST" ) \
      || { echo "❌ $tag step $s 合并失败"; exit 1; }
    echo "    ✅ $(du -sh "$DST" | cut -f1)"
  done
done

# ---------- 2a. TreeBench: infer_privilege.py 本地 HF 推理 (无需服务) ----------
# 口径 = base 44.94 那把尺子: --privilege none --no-think, 首字母规则判分。
# 输出名: {模型目录名小写}-priv-none-nothink_answer.jsonl
if [[ " $BENCHES " == *" treebench "* ]]; then
  for tag in $TAGS; do
    for s in $STEPS; do
      MODEL_DIR="$MERGE_ROOT/$tag-step$s"
      echo; echo ">>> TreeBench: $tag step $s"
      ( cd "$VOPD_ROOT/eval/treebench_probe" && \
        "$VOPD_PY" infer_privilege.py --model "$MODEL_DIR" --privilege none --no-think ) \
        || echo "❌ treebench $tag step $s 失败, 继续"
    done
  done
fi

# ---------- 2b. V*: serve + run_eval.sh (07_eval 管线, NO_JUDGE) ----------
if [[ " $BENCHES " == *" vstar "* ]]; then
  for tag in $TAGS; do
    sn=$(short_name "$tag")
    for s in $STEPS; do
      name="$sn-step$s"
      echo; echo "##########################################################"
      echo "# vstar: $name  <-  $MERGE_ROOT/$tag-step$s"
      echo "##########################################################"
      bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1; sleep 5
      MODEL="$MERGE_ROOT/$tag-step$s" SERVED_NAME="$name" bash "$SETUP/06_serve.sh" start \
        || { echo "❌ $name 起服务失败, 跳过"; continue; }
      OPENAI_MODEL_ID="$name" NO_JUDGE=1 PARALLEL_WORKERS=32 \
        bash "$SETUP/07_eval.sh" vstar || echo "❌ vstar $name 判分步失败(NO_JUDGE 下预期)"
      bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1; sleep 5
    done
  done
fi

# ---------- 3. 汇总 ----------
echo; echo "=========================================================="
echo " TreeBench 汇总 (首字母规则, 严格下界; base-4B 同口径 = 44.94)"
echo "=========================================================="
"$VOPD_PY" - "$VOPD_ROOT/eval/model_answer/treebench" <<'PY'
import json, sys, pathlib, collections
d = pathlib.Path(sys.argv[1])
for f in sorted(d.glob("*step*-priv-none-nothink_answer.jsonl")):
    recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    ok = tot = 0
    bycat = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        cat = r.get("category", "?").split("/")[-1]
        pred = str(r.get("prediction", "")).strip().upper()[:1]
        ans = str(r.get("answer", "")).strip().upper()[:1]
        tot += 1; bycat[cat][1] += 1
        if pred and pred == ans:
            ok += 1; bycat[cat][0] += 1
    print(f"  {f.name}: {ok}/{tot} = {100*ok/max(tot,1):.2f}%")
    print("    " + "  ".join(f"{c}:{100*v[0]/max(v[1],1):.1f}" for c, v in sorted(bycat.items())))
PY
echo; echo "=========================================================="
echo " V* 汇总 (16_score_norule, 与 32 号同尺; base-rerun 见旧日志)"
echo "=========================================================="
for f in "$VOPD_ROOT/eval/model_answer/vstar"/{sa4b,v04b}-*_answer.jsonl; do
  [ -e "$f" ] || continue
  m=$(basename "$f" _answer.jsonl)
  echo "--- $m ---"
  "$VOPD_PY" "$SETUP/16_score_norule.py" --benchmark vstar --model "$m" --by-category 2>&1 | sed 's/^/    /'
done
echo; echo " 结束 $(date)"
