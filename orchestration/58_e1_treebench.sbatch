#!/bin/bash
# e1 (Vision-OPD-6K 高清搜索数据训练的 OPD) checkpoint 的 TreeBench 迁移评测。
# 判决"同视觉体制(3337 token)训练的 OPD 能否迁移到 TreeBench"——数据论的终审。
# 注意: e1 = xy_rb 坐标输出变体, 格式漂移使规则口径失真, 以 judge 口径为准。
#   sbatch 58_e1_treebench.sbatch
#SBATCH --job-name=e1-treebench
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/e1tb-%j.log
#SBATCH --open-mode=append
set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
MERGE=$VOPD_ROOT/merged
JPORT=8813
JUDGE_MODEL=openai/gpt-oss-120b
VLLM_BIN="$(dirname "$VOPD_PY")/vllm"

echo "== e1 TreeBench 迁移评测  $(date)  节点=$(hostname)"

# ---------- 1. 推理 (3 ckpt, 双卡: 2并行+1) ----------
cd "$VOPD_ROOT/eval/treebench_probe"
CUDA_VISIBLE_DEVICES=0 "$VOPD_PY" infer_privilege.py \
  --model "$MERGE/vopd-xy_rb-e1-step130" --privilege none --no-think &
P0=$!
CUDA_VISIBLE_DEVICES=1 "$VOPD_PY" infer_privilege.py \
  --model "$MERGE/vopd-xy_rb-e1-step195" --privilege none --no-think &
P1=$!
wait $P0 || echo "⚠️ step130 非零退出"
wait $P1 || echo "⚠️ step195 非零退出"
CUDA_VISIBLE_DEVICES=0 "$VOPD_PY" infer_privilege.py \
  --model "$MERGE/vopd-xy_rb-e1-step260" --privilege none --no-think \
  || echo "⚠️ step260 非零退出"
echo ">>> 推理完成 $(date)"

# ---------- 2. gpt-oss 判卷 ----------
"$VLLM_BIN" serve "$JUDGE_MODEL" --port "$JPORT" \
  --gpu-memory-utilization 0.92 --max-model-len 8192 \
  > "$SETUP/logs/gptoss-e1tb-${SLURM_JOB_ID:-x}.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
ok=0
for i in $(seq 1 150); do
  sleep 10
  kill -0 $SRV 2>/dev/null || { echo "❌ judge 退出"; tail -20 "$SETUP/logs/gptoss-e1tb-${SLURM_JOB_ID:-x}.log"; exit 1; }
  curl -sf "http://localhost:$JPORT/v1/models" >/dev/null 2>&1 && { ok=1; break; }
done
[ $ok -eq 1 ] || { echo "❌ judge 未就绪"; exit 1; }
echo "✅ judge 就绪"

"$VOPD_PY" judge_treebench.py \
  --api-base "http://localhost:$JPORT/v1/" --judge-model "$JUDGE_MODEL" \
  --models \
    vopd-xy_rb-e1-step130-priv-none-nothink \
    vopd-xy_rb-e1-step195-priv-none-nothink \
    vopd-xy_rb-e1-step260-priv-none-nothink
kill $SRV 2>/dev/null; trap - EXIT
echo "== 参照: base 46.91 / V0-region峰 48.40 (judge口径)"
echo "结束 $(date)"
