#!/bin/bash
# 接力脚本: 等 interactive 上当前 infer 进程结束, 依次跑第二轮四臂
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
cd "$VOPD_ROOT/eval/treebench_probe"
LB="$VOPD_ROOT/eval/treebench_gt_labels.jsonl"
# 等当前进程(用户的 nosent run)结束
while pgrep -f "infer_privilege.py|infer_staged.py" | grep -qv $$; do sleep 60; done
echo "=== relay start $(date) ==="
"$VOPD_PY" infer_privilege.py --model Qwen/Qwen3.5-9B --privilege hide_full --no-think 2>&1 | tee q35_9b_hidefull_nothink.log
"$VOPD_PY" infer_privilege.py --model Qwen/Qwen3.5-4B --privilege draw --labels-from "$LB" --no-priv-sentence --no-think 2>&1 | tee q35_4b_draw_nosent.log
"$VOPD_PY" infer_staged.py --model Qwen/Qwen3.5-4B --condition boxfirst --no-prefill --no-system --no-think --max-new-tokens 1536 2>&1 | tee q35_4b_boxfirst_nothink.log
"$VOPD_PY" infer_staged.py --model Qwen/Qwen3.5-9B --condition boxfirst --no-prefill --no-system --no-think --max-new-tokens 1536 2>&1 | tee q35_9b_boxfirst_nothink.log
echo "=== relay done $(date) ==="
