#!/bin/bash
# 盯一个训练作业, 每 10 分钟记一次指标, 每 25 步跑一次 rollout 分析。
#   setsid nohup bash 30_monitor.sh <jobid> >/dev/null 2>&1 &
# 脱离会话运行 —— 上一轮的监视是普通后台进程, 会话一结束就被回收了。
set -uo pipefail
J="${1:?用法: 30_monitor.sh <jobid>}"
SETUP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SETUP/00_env.sh" >/dev/null 2>&1
LOG="$SETUP/logs/monitor-$J.log"
JOBLOG="$SETUP/logs/vopd-$J.log"
ROLLOUT="$VOPD_ROOT/rollouts/vopd-xy_rb"
DATA="$VOPD_ROOT/data/train_xy_rb.parquet"
say(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

say "开始监视 job $J"
last_analyzed=0
while true; do
  state=$(squeue -j "$J" -h -o "%T" 2>/dev/null)
  if [ -z "$state" ]; then
    say "作业已离开队列 —— $(sacct -j "$J" --format=State,Elapsed,MaxRSS -n -a 2>/dev/null | awk 'NF' | head -2 | tr '\n' ' ')"
    say "最终分析:"
    "$VOPD_PY" "$SETUP/28_analyze_rollouts.py" "$ROLLOUT" "$DATA" >> "$LOG" 2>&1
    break
  fi
  if [ "$state" != "RUNNING" ]; then say "状态 $state"; sleep 300; continue; fi

  # 最近一步的关键指标
  line=$(grep -oE "step:[0-9]+ - .*" "$JOBLOG" 2>/dev/null | tail -1)
  step=$(echo "$line" | grep -oE "^step:[0-9]+" | cut -d: -f2)
  m=$(echo "$line" | grep -oE "raw_jsd_token_mean:[0-9.e-]+|actor/grad_norm:[0-9.e-]+|timing_s/step:[0-9.]+|response_length/mean:[0-9.]+" | tr '\n' ' ')
  rss=$(sstat -j "$J" --format=MaxRSS -a -n 2>/dev/null | tr -d ' ' | head -1)
  say "step ${step:-?}/120  $m RSS=$rss"

  # 每 25 步做一次泄漏检查 —— v2 是在 step 50 突然崩的, 前 50 步是关键窗口
  if [ -n "${step:-}" ] && [ "$step" -ge $((last_analyzed+25)) ]; then
    last_analyzed=$step
    say "--- step $step 的 rollout 分析 ---"
    "$VOPD_PY" "$SETUP/28_analyze_rollouts.py" "$ROLLOUT" "$DATA" 2>&1 \
      | grep -E "^ *[0-9]+ " | tail -6 >> "$LOG"
    # 复读率或答题率异常就大声记一笔
    bad=$("$VOPD_PY" "$SETUP/28_analyze_rollouts.py" "$ROLLOUT" "$DATA" 2>/dev/null \
      | awk '/^ *[0-9]+ /{ans=$3+0; par=$5+0} END{if(ans<85||par>5) print "ALERT ans="ans" parrot="par}')
    [ -n "$bad" ] && say "🔴 $bad  —— 答题率<85% 或复读>5%, 可能在重演 v2 的崩溃"
  fi
  sleep 600
done
say "监视结束"
