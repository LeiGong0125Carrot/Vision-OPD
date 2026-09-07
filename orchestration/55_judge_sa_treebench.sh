#!/bin/bash
# TreeBench 官方口径判卷: base + 6 checkpoint (+4B hide无句参照)。
# 用法(GPU interactive session): bash 55_judge_sa_treebench.sh
set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
PORT=8813
JUDGE_MODEL=openai/gpt-oss-120b
VLLM_BIN="$(dirname "$VOPD_PY")/vllm"
SRVLOG="$SETUP/logs/gptoss-serve-tb-$$.log"

if ! curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
  echo ">>> 起 judge 服务 (缓存已热, 预计几分钟)..."
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
  [ $ok -eq 1 ] || { echo "❌ 服务 25 分钟没就绪"; exit 1; }
  echo "✅ judge 服务就绪 ($((i*10))s)"
else
  echo "✅ 复用已在运行的 judge 服务"
fi

cd "$VOPD_ROOT/eval/treebench_probe"
"$VOPD_PY" judge_treebench.py \
  --api-base "http://localhost:$PORT/v1/" --judge-model "$JUDGE_MODEL" \
  --models \
    qwen3.5-4b-direct-nothink \
    qwen3.5-4b-priv-hide-nosent-nothink \
    sa-opd-qwen3.5-4b-step14-priv-none-nothink \
    sa-opd-qwen3.5-4b-step28-priv-none-nothink \
    sa-opd-qwen3.5-4b-step41-priv-none-nothink \
    stdopd-region-qwen3.5-4b-step14-priv-none-nothink \
    stdopd-region-qwen3.5-4b-step28-priv-none-nothink \
    stdopd-region-qwen3.5-4b-step41-priv-none-nothink
echo "结束 $(date)"
