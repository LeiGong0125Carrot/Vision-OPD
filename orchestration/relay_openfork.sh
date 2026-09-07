#!/bin/bash
# 接力: 等 9B openfork 结束后自动跑 4B openfork
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
cd "$VOPD_ROOT/eval/treebench_probe"
while pgrep -f "probe_opening_fork.py" | grep -qv $$; do sleep 60; done
echo "=== relay: 9B done, starting 4B openfork $(date) ==="
"$VOPD_PY" probe_opening_fork.py --model Qwen/Qwen3.5-4B 2>&1 | tee q35_4b_openfork.log
echo "=== relay done $(date) ==="
