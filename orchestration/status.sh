#!/bin/bash
# 随时查看两个训练作业的状态。不依赖后台进程 —— 后台监视会随 Claude 会话退出而消失,
# 这个脚本任何时候手动跑都行。
#   bash status.sh
S=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
CK=/sfs/weka/scratch/nkw3mr/Vision-OPD/checkpoints
printf '\n===== %s =====\n' "$(date '+%m-%d %H:%M')"
squeue -u nkw3mr -o "%.10i %.10j %.4t %.9M %.10L %.13R" 2>/dev/null

for L in "$S"/logs/vopd-*-*.log; do
  [ -f "$L" ] || continue
  J=$(basename "$L" | grep -oE '[0-9]{6,}')
  [ -n "$J" ] || continue
  # 跳过启动即失败的(没跑出过 step)
  last=$(grep -oE "step:[0-9]+ - " "$L" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
  [ -n "$last" ] || continue
  tot=$(grep -oE "限制 [0-9]+ 步" "$L" | tail -1 | grep -oE '[0-9]+')
  st=$(sacct -j "$J" --format=State -n -P 2>/dev/null | head -1)
  rss=$(sacct -j "$J" --format=MaxRSS -n -P 2>/dev/null | sed -n 2p)
  [ -z "$rss" ] && rss=$(sstat -j "$J" --format=MaxRSS -a -n 2>/dev/null | tr -d ' ' | head -1)
  gib=$(echo "$rss" | tr -d 'K' | awk '{if($1!="")printf "%.1f GiB (%.0f%% of 512)", $1/1048576, $1/5368709.12}')
  line=$(grep -oE "step:$last - .*" "$L" | tail -1)
  printf '\n%s  [%s]  step %s/%s\n' "$J" "${st:-?}" "$last" "${tot:-?}"
  printf '  步时 %s   JSD %s   grad_norm %s\n' \
    "$(echo "$line"|grep -oE 'timing_s/step:[0-9.]+'|cut -d: -f2|cut -c1-6)" \
    "$(echo "$line"|grep -oE 'raw_jsd_token_mean:[0-9.e-]+'|cut -d: -f2|cut -c1-8)" \
    "$(echo "$line"|grep -oE 'actor/grad_norm:[0-9.e-]+'|cut -d: -f2|cut -c1-6)"
  printf '  内存 %s\n' "${gib:-?}"
  d=$(grep -oE "'default_local_dir': '[^']*'" "$L" 2>/dev/null | head -1 | sed "s/.*: '//; s/'$//")
  [ -n "$d" ] && printf '  ckpt %s  (%s)\n' "$(ls -1 "$d" 2>/dev/null|grep global_step|tr '\n' ' ')" "$(du -sh "$d" 2>/dev/null|cut -f1)"
  err=$(grep -cE "CUDA out of memory|OutOfMemoryError|AcceleratorError" "$L" 2>/dev/null)
  [ "${err:-0}" -gt 0 ] && printf '  ❌ 检测到 %s 处致命错误\n' "$err"
done
echo
