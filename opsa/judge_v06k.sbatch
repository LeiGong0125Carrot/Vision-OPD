#!/bin/bash
# 补判卷: V0-6karmA-pre 的 11 个 TreeBench 答案文件 (从未过 LLM judge, 公平对比需要)
#   sbatch opsa/judge_v06k.sbatch
#SBATCH --job-name=judge-v06k
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/judge-v06k-%j.log
set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
JPORT=8813
JUDGE_MODEL=openai/gpt-oss-120b
VLLM_BIN="$(dirname "$VOPD_PY")/vllm"

"$VLLM_BIN" serve "$JUDGE_MODEL" --port "$JPORT" \
  --gpu-memory-utilization 0.92 --max-model-len 8192 \
  > "$SETUP/logs/gptoss-judge-v06k-${SLURM_JOB_ID:-x}.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
ok=0
for i in $(seq 1 150); do
  sleep 10
  kill -0 $SRV 2>/dev/null || { echo "judge 服务退出"; exit 1; }
  curl -sf "http://localhost:$JPORT/v1/models" >/dev/null 2>&1 && { ok=1; break; }
done
[ $ok -eq 1 ] || { echo "judge 25min 未就绪"; exit 1; }
echo "judge 就绪 $(date)"

cd "$VOPD_ROOT/eval/treebench_probe"
TBM=""
for s in 5 10 15 20 25 30 35 40 45 50 51; do
  TBM="$TBM stdopd-region-qwen3.5-4b-6karma-pre-step${s}-priv-none-nothink"
done
"$VOPD_PY" judge_treebench.py \
  --api-base "http://localhost:$JPORT/v1/" --judge-model "$JUDGE_MODEL" --models $TBM
kill $SRV 2>/dev/null; trap - EXIT
echo "完成 $(date)"
