#!/bin/bash
# 守候某个 pending 的 Slurm 作业, 一旦分配到节点就自动挂上 GPU keepalive。
#
#   bash watch_and_keepalive.sh <jobid> [max_hours]
#
# 为什么不能简单地 srun + nohup:
#   srun 起的是一个 job step, step 结束时 Slurm 会按 cgroup 杀掉里面所有进程 ——
#   nohup 也救不了。所以这里让 srun **前台**跑一个 wrapper, wrapper 把 keepalive
#   放到后台并 `wait` 住, srun 因此一直存活, keepalive 也就活着。
#
# 同时 wrapper 会把 keepalive 的 PID 写进 .keepalive.pid, 这样:
#   - 用户回来后 `bash keepalive.sh stop` 能正常停掉它
#   - `09_train_smoke.sh` 开跑前会读这个文件自动停掉 keepalive(否则它会占着显存抢时间片)
set -uo pipefail

SETUP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JID="${1:?用法: bash watch_and_keepalive.sh <jobid> [max_hours]}"
MAXH="${2:-5}"                 # keepalive 自动退出小时数(兜底, 作业本身一般先到点)
LOG="$SETUP/watch_keepalive.log"
POLL=30                        # 轮询间隔(秒)
GIVEUP=$((12 * 3600))          # 最多守 12 小时

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "开始守候 job $JID (每 ${POLL}s 轮询, 最多 12h)"

waited=0
while :; do
  st=$(squeue -j "$JID" -h -o "%t" 2>/dev/null | tr -d ' ')
  if [ -z "$st" ]; then
    say "job $JID 已不在队列(被取消或已结束), 退出守候"; exit 1
  fi
  if [ "$st" = "R" ]; then
    node=$(squeue -j "$JID" -h -o "%N" | tr -d ' ')
    say "job $JID 已分配到 $node, 准备挂 keepalive"
    break
  fi
  # 每 10 分钟报一次, 免得日志太吵
  if [ $((waited % 600)) -eq 0 ]; then
    say "状态=$st, 已等 $((waited / 60)) 分钟"
  fi
  sleep "$POLL"
  waited=$((waited + POLL))
  if [ "$waited" -ge "$GIVEUP" ]; then say "等满 12h 仍未分配, 放弃"; exit 1; fi
done

# 已经有一个在跑就别重复挂
if [ -f "$SETUP/.keepalive.pid" ] && kill -0 "$(cat "$SETUP/.keepalive.pid")" 2>/dev/null; then
  say "keepalive 已在运行 (pid=$(cat "$SETUP/.keepalive.pid")), 不重复启动"; exit 0
fi

say "srun --overlap 进入 job $JID 启动 keepalive (burst=25s idle=5s 自动退出=${MAXH}h)"
srun --overlap --jobid="$JID" --ntasks=1 bash -c "
  cd '$SETUP' || exit 1
  echo \"[\$(date '+%m-%d %H:%M:%S')] keepalive 启动于 \$(hostname), 可见 GPU: \$(nvidia-smi -L 2>/dev/null | wc -l)\" >> '$LOG'
  ./gpu_keepalive 25 5 $MAXH >> '$SETUP/keepalive.log' 2>&1 &
  child=\$!
  echo \$child > '$SETUP/.keepalive.pid'
  echo \"[\$(date '+%m-%d %H:%M:%S')] keepalive pid=\$child, 已写入 .keepalive.pid\" >> '$LOG'
  wait \$child
  rm -f '$SETUP/.keepalive.pid'
" >> "$LOG" 2>&1

say "keepalive 结束 (作业到点 / 被 keepalive.sh stop 停掉 / 被训练脚本自动停掉)"
