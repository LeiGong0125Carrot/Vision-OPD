#!/bin/bash
# GPU keep-alive 控制脚本
#   bash keepalive.sh start [burst_s] [idle_s] [max_hours]
#   bash keepalive.sh status | stop | log
# 必须在 salloc 拿到的那个 B200 节点的 shell 里跑 (进程随 job 存活)。
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$DIR/gpu_keepalive"
SRC="$DIR/gpu_keepalive.cu"
PIDF="$DIR/.keepalive.pid"
LOG="$DIR/keepalive.log"

build() {
  command -v nvcc >/dev/null || { module purge; module load cuda/12.8.0; }
  command -v nvcc >/dev/null || { echo "找不到 nvcc, 先 module load cuda/12.8.0"; exit 1; }
  [ -f "$BIN" ] && [ "$BIN" -nt "$SRC" ] && return 0
  echo ">>> 编译 gpu_keepalive (nvcc $(nvcc --version | grep -o 'release [0-9.]*'))"
  # B200 = sm_100; 带 PTX 以便在别的卡上也能 JIT
  nvcc -O2 -o "$BIN" "$SRC" \
       -gencode arch=compute_100,code=sm_100 \
       -gencode arch=compute_100,code=compute_100 2>/dev/null \
    || nvcc -O2 -o "$BIN" "$SRC" -arch=native 2>/dev/null \
    || nvcc -O2 -o "$BIN" "$SRC" \
    || { echo "编译失败"; exit 1; }
  echo ">>> OK: $BIN"
}

running() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

case "${1:-status}" in
  start)
    if running; then echo "已在运行, pid=$(cat "$PIDF")"; exit 0; fi
    build
    B="${2:-25}"; I="${3:-5}"; H="${4:-12}"
    nohup "$BIN" "$B" "$I" "$H" >> "$LOG" 2>&1 &
    echo $! > "$PIDF"
    sleep 3
    if running; then
      echo ">>> 已启动 pid=$(cat "$PIDF")  burst=${B}s idle=${I}s 自动退出=${H}h"
      echo ">>> 日志: $LOG"
      tail -5 "$LOG"
    else
      echo ">>> 启动失败, 看日志:"; tail -20 "$LOG"; rm -f "$PIDF"; exit 1
    fi
    ;;
  stop)
    if running; then
      P=$(cat "$PIDF"); kill "$P" 2>/dev/null
      for _ in $(seq 10); do kill -0 "$P" 2>/dev/null || break; sleep 1; done
      kill -9 "$P" 2>/dev/null
      echo ">>> 已停止 pid=$P"
    else
      echo ">>> 未在运行"
    fi
    rm -f "$PIDF"
    ;;
  status)
    if running; then echo ">>> 运行中 pid=$(cat "$PIDF")"; else echo ">>> 未运行"; fi
    echo; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,power.draw --format=csv 2>/dev/null \
      || echo "(此节点没有 GPU)"
    ;;
  log) tail -f "$LOG" ;;
  *) sed -n '2,6p' "$0" ;;
esac
