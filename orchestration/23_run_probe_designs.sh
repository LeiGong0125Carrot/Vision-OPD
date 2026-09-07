#!/bin/bash
# 跑候选输出格式的筛选探针(22_probe_designs.py)。
#
#   bash 23_run_probe_designs.sh              # 默认 100 题 × 8 采样 × 7 设计
#   N=200 bash 23_run_probe_designs.sh
#   DESIGNS=T_sent_copy,T_json_xy bash 23_run_probe_designs.sh   # 只跑其中几个
#
# 用 base 模型(Qwen3.5-4B)当 teacher 的代理 —— 训练开始时 EMA teacher 就是它。
# 服务必须带上**训练用的 chat template**(06_serve.sh 默认就是), 否则模型会进入
# 思考模式, 行为和训练时完全不同。
set -uo pipefail
SETUP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SETUP/00_env.sh"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
SERVED="${SERVED:-qwen35-4b}"
PORT="${PORT:-8766}"
N="${N:-100}"
SAMPLES="${SAMPLES:-8}"
OUT="${OUT:-$VOPD_ROOT/eval/probe_designs.json}"

fail(){ echo "❌ $*"; exit 1; }

# --- GPU 必须是空的。06_serve.sh stop 杀不掉子进程 VLLM::EngineCore(踩过) ---
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F', ' '$2>2000{print "  GPU "$1": "$2" MiB"}')
if [ -n "$busy" ]; then
  echo "⚠️  GPU 上有东西占着显存:"; echo "$busy"
  echo "   若是上次的残留: nvidia-smi 查 PID 后 kill -9"
  echo "   (探针只用 1 张卡, 另一张被占也能跑, 3 秒后继续)"; sleep 3
fi

echo "=========================================================="
echo " 候选输出格式筛选探针"
echo " 模型: $MODEL  端口: $PORT  规模: $N 题 × $SAMPLES 采样"
echo "=========================================================="

# --- 起服务 ---
# ⚠️ 不要用 `06_serve.sh status` 当布尔判断: 它不设退出码(最后一条是 nvidia-smi),
#    几乎永远返回 0 -> 会误判成"已在运行"。直接问端口要模型名。
ask_port(){ curl -sf "http://localhost:$1/v1/models" 2>/dev/null \
            | "$VOPD_PY" -c "import json,sys;print(' '.join(m['id'] for m in json.load(sys.stdin)['data']))" 2>/dev/null; }

have=$(ask_port "$PORT")
if [ -n "$have" ]; then
  case " $have " in
    *" $SERVED "*) echo ">>> $PORT 上已经在提供 $SERVED，直接用"; NEED_START=0 ;;
    *) echo ">>> 端口 $PORT 被占用了，上面提供的是: $have"
       # GPU 节点端口是全节点共享的(同节点其他用户也会占)，自动往后找一个空的
       for p in $(seq $((PORT+1)) $((PORT+12))); do
         if [ -z "$(ask_port "$p")" ] && ! (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null; then
           PORT=$p; break
         fi
       done
       [ -n "$(ask_port "$PORT")" ] && fail "$((PORT-12))..$PORT 全被占了，手动指定 PORT=<空闲端口>"
       echo ">>> 改用空闲端口 $PORT"; NEED_START=1 ;;
  esac
else
  NEED_START=1
fi

if [ "${NEED_START:-1}" = "1" ]; then
  echo ">>> 在端口 $PORT 启动 vLLM (首次约 3-5 分钟)..."
  MODEL="$MODEL" SERVED_NAME="$SERVED" PORT="$PORT" GPUUTIL=0.85 \
    bash "$SETUP/06_serve.sh" start || fail "服务起不来, 看 $SETUP/serve.$PORT.log"
fi

have=$(ask_port "$PORT")
case " $have " in
  *" $SERVED "*) ;;
  *) fail "端口 $PORT 上是 [$have]，不是 $SERVED" ;;
esac
echo ">>> 服务就绪 (端口 $PORT, 模型 $SERVED)"; echo

# --- 跑探针 ---
"$VOPD_PY" "$SETUP/22_probe_designs.py" \
    --model "$SERVED" --api-base "http://localhost:$PORT/v1" \
    --n "$N" --samples "$SAMPLES" --out "$OUT" \
    ${DESIGNS:+--designs "$DESIGNS"}
rc=$?

echo "=========================================================="
echo " 完成 (exit $rc)。服务还开着, 要停:"
echo "   PORT=$PORT bash $SETUP/06_serve.sh stop"
echo "=========================================================="
exit $rc
