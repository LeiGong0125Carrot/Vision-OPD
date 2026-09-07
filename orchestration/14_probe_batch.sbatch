#!/bin/bash
# 探测 512G 下能撑多大 batch —— 只跑 1 步, 测峰值内存, 不是训练。
#
#   sbatch --export=ALL,PROBE_BATCH=96 14_probe_batch.sbatch     # 论文配置 96x8=768 条
#   sbatch --export=ALL,PROBE_BATCH=48 14_probe_batch.sbatch     # 退一档 384 条
#
# 背景: 正式跑用 BATCH=24/RN=8(192 条), 干净作业实测 MaxRSS 227.4 GiB。想知道
# 512G 的上限在哪, 才能判断能不能对齐论文的 BATCH=96。
#
# 为什么要单独测: MaxRSS 是**整个作业**的历史高水位, 同一作业里跑过的失败尝试会
# 污染读数(之前 387 GiB 那个数就是这么来的)。每个探测点必须是全新作业。
#
#SBATCH --job-name=vopd-probe
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=01:30:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/probe-b%x-%j.log

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
mkdir -p "$SETUP/logs"
source "$SETUP/00_env.sh"

B="${PROBE_BATCH:-96}"
R="${PROBE_RN:-8}"
echo "=========================================================="
echo " 探测: BATCH=$B RN=$R  ->  $((B*R)) 条序列/步   (--mem=512G, 2 卡)"
echo " job $SLURM_JOB_ID on $(hostname)   $(date)"
echo "=========================================================="

# 丢弃式目录: 1 步跑完 verl 仍会存一次 checkpoint(约 53GB), 别污染正式跑的目录
THROW="$VOPD_ROOT/checkpoints/_probe_$SLURM_JOB_ID"
mkdir -p "$THROW"

# 后台采样 MaxRSS —— sstat 只在作业运行期间可查, 结束后就没了
( while :; do
    sstat -j "$SLURM_JOB_ID" --format=MaxRSS -a -n 2>/dev/null | tr -d ' ' | head -1 \
      | awk -v t="$(date +%H:%M:%S)" '{if($1!="")printf "  [%s] MaxRSS=%s (%.1f GiB)\n", t, $1, $1/1048576}'
    sleep 60
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

TRAIN_FILE="$VOPD_ROOT/data/train_bbox_text.parquet" \
BATCH="$B" RN="$R" ROLLOUT_GPU_MEM=0.45 NGPUS=2 STEPS=1 \
bash "$SETUP/09_train_smoke.sh" \
    trainer.experiment_name="probe-b$B" \
    trainer.default_local_dir="$THROW" \
    trainer.rollout_data_dir="$THROW/rollouts"
rc=$?

kill $SAMPLER 2>/dev/null
echo
echo "=========================================================="
echo " BATCH=$B RN=$R  ($((B*R)) 条)   exit=$rc"
echo " 峰值 MaxRSS: $(sstat -j "$SLURM_JOB_ID" --format=MaxRSS -a -n 2>/dev/null | tr -d ' ' | head -1)"
[ "$rc" -ne 0 ] && echo " ❌ 失败 —— 若是 OOM 说明 512G 撑不住这个 batch"
echo " 清理: rm -rf $THROW"
echo "=========================================================="
exit $rc
