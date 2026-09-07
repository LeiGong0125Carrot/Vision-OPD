#!/bin/bash
# Vision-OPD 「语言特权」版 —— BATCH=48 对照组 (对比 13_train_full.sbatch 的 BATCH=24)
#
#   sbatch 15_train_batch48.sbatch
#
# 和 13 的唯一区别是 batch: 48×8 = 384 条序列/步 (13 是 24×8 = 192 条)。
# 数据、模型、lr、alpha、topk、teacher 设置全部相同。
#
# 两个目的:
#   1) 拿到第二个**干净**的内存数据点。已有 192 条 -> 227.4 GiB (job 18809274),
#      两点定出斜率后, 才能可靠地判断 512G 到底能撑到多大 batch。
#      (注意 MaxRSS 是整作业历史高水位, 同一作业里的失败尝试会污染读数 ——
#       之前那个 387 GiB 就是这么来的, 真实值是 227。所以必须是全新作业。)
#   2) 看 batch 翻倍对收敛的影响: 梯度噪声更小, 但梯度步数减半 (130 vs 260)。
#
#SBATCH --job-name=vopd-b48
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/vopd-b48-%j.log
#SBATCH --open-mode=append

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
mkdir -p "$SETUP/logs"

echo "=========================================================="
echo " job      : ${SLURM_JOB_ID:-NA} on $(hostname)"
echo " start    : $(date)"
echo " 配置     : BATCH=48 RN=8 -> 384 条序列/步, 130 步 = 1 epoch"
echo "=========================================================="

source "$SETUP/00_env.sh"

export TRAIN_FILE="$VOPD_ROOT/data/train_bbox_text.parquet"
export BATCH=48                       # 13 是 24
export RN=8
export ROLLOUT_GPU_MEM=0.45
export NGPUS=2
export STEPS=130                      # 6241 ÷ 48 = 130 步 = 1 epoch (13 是 260 步)

CKPT_DIR="$VOPD_ROOT/checkpoints/vopd-bboxtext-b48"
ROLLOUT_DIR="$VOPD_ROOT/rollouts/vopd-bboxtext-b48"
mkdir -p "$CKPT_DIR" "$ROLLOUT_DIR"

# --- 前置检查 ---
fail() { echo "❌ $*"; exit 1; }
[ -f "$TRAIN_FILE" ] || fail "训练数据不存在: $TRAIN_FILE"
n=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || true)
[ "$n" = "$NGPUS" ] || fail "拿到 $n 张卡, 配置按 $NGPUS 张调的"
want=$(cat "$VOPD_ENV/.flash_attn_archs" 2>/dev/null)
have=$(cuobjdump --list-elf "$VOPD_ENV"/lib/python3.12/site-packages/flash_attn_2_cuda*.so 2>/dev/null \
       | grep -oE "sm_[0-9]+" | sort -u | tr '\n' ' ')
echo ">>> GPU        : $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
echo ">>> flash-attn : 编译为 [$have], 期望含 sm_$want"
case "$have" in *"sm_$want"*) ;; *) fail "flash-attn arch 不匹配当前卡" ;; esac
[ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ] && \
  echo ">>> ⚠️  已有 checkpoint (step $(cat "$CKPT_DIR/latest_checkpointed_iteration.txt")), resume_mode=auto 会续训"
echo

# 每 5 分钟采样一次 MaxRSS —— sstat 只在作业运行期间可查, 结束后就没了。
# 这次跑的核心产出之一就是这条内存曲线, 不能只靠结尾采一次。
( while :; do
    sstat -j "${SLURM_JOB_ID:-0}" --format=MaxRSS -a -n 2>/dev/null | tr -d ' ' | head -1 \
      | awk -v t="$(date '+%m-%d %H:%M')" '{if($1!="")printf "[MEM %s] MaxRSS=%s (%.1f GiB)\n", t, $1, $1/1048576}'
    sleep 300
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

# lr_warmup_steps 按「warmup 期间看到的数据量」和论文对齐:
#   论文 10 步 × 768 条 = 7680 条
#   本配置 384 条/步 -> 20 步   (13 那个是 192 条/步 -> 40 步)
# 这样两个跑之间、以及和论文之间, warmup 阶段喂进去的数据量都一致。
# lr 保持 2e-6 不变, 和 13 一致 —— 两个跑之间只差 batch 这一个变量。
#
# save_freq=25: 130 ÷ 25 ≈ 5 次, 和 13 的 260÷50 存盘次数相同, 便于对比开销。
# !! 不能直接调 09_train_smoke.sh !!
# 它把三个路径写死了, 而这三个都会和正在跑的 job 18809274 撞车:
#   DST=$SETUP/run_vision_opd_smoke.sh   两边都用 sed 重新生成同一个文件
#   LOG=$SETUP/train_smoke.log           两边都 tee 到同一个文件(后来者会截断它)
#   RAY_TMPDIR=$TMPDIR/ray               两个 Ray 实例共用临时根目录
#
# 其中第一个最危险: bash 是**增量读取**脚本的, job 1 的 bash 此刻正持有
# run_vision_opd_smoke.sh 并记着字节偏移(卡在最后那条 python 上, 要等 17 小时),
# 我们一旦覆盖它, job 1 在 python 退出后就会从错位的位置继续读 -> 执行到垃圾。
#
# 同理也**不能直接编辑 09_train_smoke.sh 本身** —— job 1 的 bash 也持有它。
# 所以这里做一份带 jobid 后缀的副本, 只改这三个路径, 原文件一个字节都不动。
#
# !! RAY_TMPDIR 必须用**短**名字, 不能带 jobid !!
# Ray 会在它下面拼 /ray/session_<26字符时间戳>_<pid>/sockets/plasma_store, 共 67 字节,
# 而 AF_UNIX 路径上限是 107 字节 -> RAY_TMPDIR 最多 40 字节。
#     /sfs/weka/scratch/nkw3mr/tmp/ray           32  ✓
#     /sfs/weka/scratch/nkw3mr/tmp/ray_18812988  41  ✗ 超 1 字节, job 18812988 就这么挂的
# 所以用固定短名 ray_b48 (36 字节) 而不是 jobid。它仍然和 job 1 的 $TMPDIR/ray 分开,
# 隔离目的照样达到 —— 同一个 sbatch 不会并发跑两份。
RAY_TMP="$TMPDIR/ray_b48"
if [ ${#RAY_TMP} -gt 40 ]; then
  fail "RAY_TMPDIR 太长(${#RAY_TMP} > 40 字节): $RAY_TMP —— Ray 的 AF_UNIX socket 会超 107 字节上限"
fi
echo ">>> RAY_TMPDIR : $RAY_TMP (${#RAY_TMP}/40 字节)"

RUNNER="$SETUP/.runner_${SLURM_JOB_ID}.sh"
sed -e "s|^DST=.*|DST=\"$SETUP/.gen_${SLURM_JOB_ID}.sh\"|" \
    -e "s|^LOG=.*|LOG=\"$SETUP/logs/train_${SLURM_JOB_ID}.log\"|" \
    -e "s|^export RAY_TMPDIR=.*|export RAY_TMPDIR=\"$RAY_TMP\"|" \
    "$SETUP/09_train_smoke.sh" > "$RUNNER"
echo ">>> 隔离检查（应全部带 jobid $SLURM_JOB_ID）:"
grep -nE '^(DST|LOG)=|^export RAY_TMPDIR=' "$RUNNER" | sed 's/^/    /'
echo

bash "$RUNNER" \
    trainer.save_freq=25 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.experiment_name=vopd-bboxtext-b48 \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.rollout_data_dir="$ROLLOUT_DIR" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20
rc=$?

kill $SAMPLER 2>/dev/null
echo
echo "=========================================================="
echo " end   : $(date)   exit=$rc"
echo " ckpt  : $(ls -1 "$CKPT_DIR" 2>/dev/null | tr '\n' ' ')"
echo " 峰值内存(MaxRSS): $(sstat -j "${SLURM_JOB_ID:-0}" --format=MaxRSS -a -n 2>/dev/null | tr -d ' ' | head -1)"
echo "=========================================================="
exit $rc
