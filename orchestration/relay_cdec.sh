#!/bin/bash
# 接力: 等用户的 a1.0 全量结束 -> a0.5 全量 -> hide 链熵剖面 -> keepalive
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
cd "$VOPD_ROOT/eval/treebench_probe"
while pgrep -f "probe_contrastive_decode|infer_privilege|infer_staged|probe_init_confidence" | grep -qv $$; do sleep 60; done
echo "=== relay start $(date) ==="
"$VOPD_PY" probe_contrastive_decode.py --model Qwen/Qwen3.5-9B --alpha 0.5 --deprived shuffled 2>&1 | tee 9B_cdec_a05.log
"$VOPD_PY" probe_init_confidence.py --model Qwen/Qwen3.5-9B --privilege hide 2>&1 | tee 9B_init_conf_hide.log
echo "=== relay done $(date), keepalive ==="
while true; do sleep 300; done
