#!/bin/bash
# 起 Vision-OPD 的 vLLM OpenAI 兼容服务。
#   bash 06_serve.sh start   [模型]     后台启动并等待就绪
#   bash 06_serve.sh status | stop | log | test
#
# 与 README 的差异（README 是 8 卡配置，直接抄会出问题）：
#   --tensor-parallel-size 1   单卡
#   --max-model-len 65536      模型 max_position_embeddings=262144(256K)，不指定的话
#                              vLLM 会照 256K 分配 KV cache，启动极慢且没必要。
#                              !! 但也不能设太小: 它是"输入+输出"的总上限, 而 run_eval.sh
#                              默认 MAX_TOKENS(输出)=32768。曾经设 32768 导致
#                              "requested 32768 output tokens ... upper bound for 0 input
#                              tokens" —— 每条请求都 400。65536 给输入留够空间。
set -uo pipefail

SETUP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SETUP/00_env.sh"
conda activate "$VOPD_ENV"

MODEL="${MODEL:-yuanqianhao/Vision-OPD-9B}"
SERVED_NAME="${SERVED_NAME:-Vision-OPD-9B}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-65536}"
GPUUTIL="${GPUUTIL:-0.85}"
# 训练用的自定义模板(run_vision_opd.sh:76)会在 assistant 头后**预填一对空的
# <think></think>**, 强制模型跳过思维链直接作答 —— 训练时回复只有 2~102 token
# 就是这么来的。要让服务端的行为和训练一致(比如跑探针实验), 必须带上它;
# 不带的话模型会进入思考模式, 输出几百 token 的推理, 完全是另一种行为。
# CHAT_TEMPLATE=none 可显式关掉。
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$VOPD_ROOT/chat_templates/perception_chat_template_qwen35.jinja}"
# 日志和 PID 文件按端口区分 —— 否则同时起两个服务(比如被测模型 8765 + judge 8766)
# 时, 第二个会因为读到第一个的 .serve.pid 而报"已在运行"直接退出。
LOG="$SETUP/serve.$PORT.log"
PIDF="$SETUP/.serve.$PORT.pid"
# 兼容早先不带端口的旧文件(默认端口 8000)
[ "$PORT" = "8000" ] && [ -f "$SETUP/.serve.pid" ] && [ ! -f "$PIDF" ] && mv "$SETUP/.serve.pid" "$PIDF"

running() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

# ready() 必须确认"端口上的服务提供的是**我们的**模型"。
# 只 curl /v1/models 通不通是不够的: GPU 节点是共享的, 别的用户可能正好也在
# 这个端口上跑 vLLM。曾经就因此被别人的 Qwen3-0.6B 骗到, 报了"✅ 就绪",
# 而我们自己的进程其实已经因 "Address already in use" 崩掉了。
ready() {
  curl -sf "http://localhost:$PORT/v1/models" 2>/dev/null \
    | grep -q "\"id\"[[:space:]]*:[[:space:]]*\"$SERVED_NAME\""
}
port_busy() { curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; }

case "${1:-status}" in
  start)
    if running; then echo "已在运行 pid=$(cat "$PIDF")"; exit 0; fi

    # 端口被别人占着的话, vllm 会直接 bind 失败退出 —— 提前说清楚, 别等 20 分钟
    if port_busy; then
      echo "❌ 端口 $PORT 上已经有服务在跑，但不是我们启动的（$PIDF 里没有活进程）："
      curl -s "http://localhost:$PORT/v1/models" \
        | "$VOPD_PY" -c "import json,sys; print('   它提供:', [m['id'] for m in json.load(sys.stdin)['data']])" 2>/dev/null
      echo "   GPU 节点是共享的，这多半是同节点其他用户的服务。"
      echo "   换个端口重来:  PORT=8765 MODEL=$MODEL SERVED_NAME=$SERVED_NAME bash 06_serve.sh start"
      exit 1
    fi

    # keepalive 的 CUDA context 会占显存、抢时间片
    if [ -f "$SETUP/.keepalive.pid" ] && kill -0 "$(cat "$SETUP/.keepalive.pid")" 2>/dev/null; then
      echo ">>> 先停掉 keepalive"; bash "$SETUP/keepalive.sh" stop; sleep 3
    fi

    [ -n "${2:-}" ] && MODEL="$2"
    echo ">>> 解释器: $VOPD_PY  ($("$VOPD_PY" -V 2>&1))"
    # MODEL 可以是 HF repo id, 也可以是本地目录(比如 model_merger 合并出来的 checkpoint)。
    # snapshot_download 只认 repo id, 传本地路径会报
    #   HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'
    # 所以先判断: 是目录就跳过下载, 并顺手检查必需文件在不在。
    if [ -d "$MODEL" ]; then
      echo ">>> 本地模型目录, 跳过下载: $MODEL"
      ls "$MODEL"/*.safetensors >/dev/null 2>&1 || { echo "❌ 目录里没有 *.safetensors —— 是不是还没合并?"; exit 1; }
      [ -f "$MODEL/config.json" ] || { echo "❌ 缺 config.json"; exit 1; }
      echo "    权重 $(du -sh "$MODEL" | cut -f1)"
    else
      echo ">>> 预下载权重（约 18GB，落在 $HF_HOME）"
      "$VOPD_PY" - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1])
print("   本地路径:", p)
PY
      [ $? -ne 0 ] && { echo "下载失败"; exit 1; }
    fi

    TPL_ARGS=()
    if [ "$CHAT_TEMPLATE" != "none" ]; then
      if [ ! -f "$CHAT_TEMPLATE" ]; then
        echo "❌ chat template 不存在: $CHAT_TEMPLATE"; exit 1
      fi
      TPL_ARGS=(--chat-template "$CHAT_TEMPLATE")
    fi

    echo
    echo ">>> 启动 vLLM  模型=$MODEL  端口=$PORT  max_len=$MAXLEN  gpu_util=$GPUUTIL"
    echo ">>> chat template: $CHAT_TEMPLATE"
    nohup "$VOPD_ENV/bin/vllm" serve "$MODEL" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "$GPUUTIL" \
        --max-model-len "$MAXLEN" \
        --served-model-name "$SERVED_NAME" \
        --port "$PORT" \
        --trust-remote-code \
        "${TPL_ARGS[@]}" \
        > "$LOG" 2>&1 &
    echo $! > "$PIDF"
    echo ">>> pid=$(cat "$PIDF")  日志: $LOG"

    echo ">>> 等待就绪（首次启动含 FlashInfer JIT + CUDA graph 捕获，可能几分钟）"
    for i in $(seq 1 120); do        # 最多等 20 分钟
      # 先看进程死没死, 再看就绪 —— 反过来的话, 崩溃瞬间若端口上恰好有别人的服务
      # 应答, 就会误报成功。
      if ! running; then
        echo; echo "❌ 进程已退出，日志末尾："; tail -30 "$LOG"; rm -f "$PIDF"; exit 1
      fi
      if ready; then
        echo; echo "✅ 服务就绪 (等了 $((i*10))s)"
        curl -s "http://localhost:$PORT/v1/models" | "$VOPD_PY" -c "import json,sys; print('   已注册模型:', [m['id'] for m in json.load(sys.stdin)['data']])"
        echo
        echo "下一步跑评测（另开一个 shell）:"
        echo "  API_BASE=\"http://localhost:$PORT/v1/\" OPENAI_MODEL_ID=\"$SERVED_NAME\" \\"
        echo "  BENCHMARK=\"vstar\" bash \$VOPD_ROOT/eval/run_eval.sh"
        exit 0
      fi
      if ! running; then
        echo; echo "❌ 进程已退出，日志末尾："; tail -30 "$LOG"; rm -f "$PIDF"; exit 1
      fi
      printf "."; sleep 10
    done
    echo; echo "⚠️  20 分钟仍未就绪，看日志: tail -f $LOG"; exit 2
    ;;

  stop)
    if running; then
      P=$(cat "$PIDF"); kill "$P" 2>/dev/null
      for _ in $(seq 20); do kill -0 "$P" 2>/dev/null || break; sleep 1; done
      kill -9 "$P" 2>/dev/null; pkill -f "vllm serve" 2>/dev/null
      echo ">>> 已停止 pid=$P"
    else echo ">>> 未在运行"; fi
    rm -f "$PIDF"
    ;;

  status)
    # 必须把查的是哪个端口/哪个模型名打出来。PORT 和 SERVED_NAME 都有默认值(8000 /
    # Vision-OPD-9B), 忘了传就会去查错的端口、找错的模型名, 然后报"未就绪" ——
    # 而服务其实好好的。踩过一次。
    echo ">>> 检查目标   : localhost:$PORT   模型名 '$SERVED_NAME'"
    echo "                 (这两个来自 PORT / SERVED_NAME 环境变量, 未传时用默认值)"
    running && echo ">>> 进程存活 pid=$(cat "$PIDF")" || echo ">>> 进程不在 (注意: PID 是按节点隔离的)"
    if ready; then
      echo ">>> HTTP 就绪"
    else
      echo ">>> HTTP 未就绪"
      # 端口上有服务但模型名对不上, 是最常见的情形, 直接把实际提供的列出来
      if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "    但端口 $PORT 上**有**服务, 它提供的是:"
        curl -s "http://localhost:$PORT/v1/models" \
          | "$VOPD_PY" -c "import json,sys; print('      ', [m['id'] for m in json.load(sys.stdin)['data']])" 2>/dev/null
        echo "    -> 用 SERVED_NAME=<上面那个> 重新 status, 或换端口另起一个"
      fi
    fi
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
    ;;

  test)
    echo ">>> 发一个纯文本请求"
    curl -s "http://localhost:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍你自己\"}],\"max_tokens\":64}" \
      | "$VOPD_PY" -m json.tool 2>/dev/null | head -30
    ;;

  log) tail -f "$LOG" ;;
  *) sed -n '2,8p' "$0" ;;
esac
