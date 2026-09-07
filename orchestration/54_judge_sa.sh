#!/bin/bash
# 交互 session 版: 只对 SA/V0 六个 checkpoint + base 校准做 gpt-oss-120b 官方判卷。
# 用法(在 GPU interactive session 里): bash 54_judge_sa.sh
set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
PORT=8813
JUDGE_MODEL=openai/gpt-oss-120b
VLLM_BIN="$(dirname "$VOPD_PY")/vllm"
EVAL_DIR="$VOPD_ROOT/eval"
TAGS="base-qwen35-4b-rerun_seed42 sa4b-step14_seed42 sa4b-step28_seed42 sa4b-step41_seed42 v04b-step14_seed42 v04b-step28_seed42 v04b-step41_seed42"
SRVLOG="$SETUP/logs/gptoss-serve-interactive-$$.log"

"$VLLM_BIN" serve "$JUDGE_MODEL" --port "$PORT" \
  --gpu-memory-utilization 0.92 --max-model-len 8192 > "$SRVLOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
ok=0
for i in $(seq 1 150); do
  sleep 10
  kill -0 $SRV 2>/dev/null || { echo "❌ vllm serve 退出:"; tail -30 "$SRVLOG"; exit 1; }
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { ok=1; break; }
done
[ $ok -eq 1 ] || { echo "❌ judge 服务 25 分钟没就绪"; exit 1; }
echo "✅ judge 服务就绪 ($((i*10))s)"

cd "$EVAL_DIR"
for tag in $TAGS; do
  [ -f "model_answer/vstar/${tag}_answer.jsonl" ] || { echo "跳过(无文件): $tag"; continue; }
  echo; echo "########## judge: $tag ##########"
  "$VOPD_PY" judge_qwenlm.py --benchmark vstar --model "$tag" \
    --api_base "http://localhost:$PORT/v1/" --judge_model "$JUDGE_MODEL"
done
kill $SRV 2>/dev/null; trap - EXIT

echo; echo "========== 官方口径汇总 (yes前缀) =========="
"$VOPD_PY" - <<PY
import json, os, re, collections
tags = "$TAGS".split()
print(f"{'模型':<30}{'judge口径':>10}{'规则下界':>10}{'LLM救回':>8}")
for tag in tags:
    p = f"judge/vstar/{tag}_answer.jsonl"
    if not os.path.exists(p): continue
    data = json.load(open(p))
    n = len(data)
    yes = sum(1 for it in data if re.match(r"^yes\b", str(it.get("judge","")).strip(), re.I))
    rule = sum(1 for it in data if it.get("judge_source") != "llm"
               and re.match(r"^yes\b", str(it.get("judge","")).strip(), re.I))
    print(f"{tag:<30}{100*yes/n:>9.2f}%{100*rule/n:>9.2f}%{yes-rule:>8}")
PY
echo "结束 $(date)"
