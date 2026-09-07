#!/bin/bash
# 训练运行器。三个变体共用, 靠 VARIANT 区分。
#
#   VARIANT=xy_rb STEPS=2 bash 26_run_train.sh    # 本地冒烟(2 步, 不存 ckpt)
#   bash 26_run_train.sh                          # 完整跑(xy_rb, 120 步)
#   sbatch 27_train.sbatch                        # 提交作业(它就是调用本脚本)
#
# 变体（均经探针 22 实测筛选, 见 25_prep_data.py 的表）:
#   xy_rb  ★ 语言特权(system 给坐标) + 红框图   师生 IoU 0.368 -> 0.803
#   xy       纯语言特权                          师生 IoU 0.368 -> 0.562
#   redbox   纯视觉特权(红框图)                  师生 IoU 0.368 -> 0.619
#
# 冒烟和正式跑**共用这一个脚本**, 只靠环境变量区分 —— 上一轮 09/15/20/21 之间复制粘贴
# 出了一堆各自的 bug(写死的路径、并发撞车、sed 规则静默失效), 不再重演。
#
# **直接调用仓库的 scripts/run_vision_opd.sh**, 不做副本、不改仓库代码。
# 要改的每一项都是 hydra override, 追加在命令行末尾即可
# (run_vision_opd.sh:164 的 EXTRA_ARGS 在最后, 重复 key 后者生效, 已验证)。
set -uo pipefail
SETUP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SETUP/00_env.sh"

VARIANT="${VARIANT:-xy_rb}"
STEPS="${STEPS:-120}"
TRAIN_FILE="${TRAIN_FILE:-$VOPD_ROOT/data/train_${VARIANT}.parquet}"

# WANT_TP  : 该变体是否该有 teacher_prompt 列(= 语言特权)
# WANT_DIFF: student 和 teacher 的图是否**应该不同**(= 视觉特权)
# 两个都要显式声明。只查其中一个的话, 另一个静默失效时实验会变成别的实验,
# 而**运行期没有任何指标能发现** —— teacher_image_swap_fraction 只说明
# bbox_images 非空(ray_trainer.py:1377 = teacher_present_mask.mean()),
# 对 teacher_prompt 用没用只字未提。开跑前这道闸门是唯一防线。
case "$VARIANT" in
  xy_rb)    WANT_TP=1; WANT_DIFF=1 ;;   # 语言 + 视觉
  xy)       WANT_TP=1; WANT_DIFF=0 ;;   # 纯语言, 两边都看干净图(所以图相同是对的)
  redbox)   WANT_TP=0; WANT_DIFF=1 ;;   # 纯视觉
  xy_rb_mb) WANT_TP=1; WANT_DIFF=1 ;;   # 同 xy_rb 特权, 多框输出格式(2026-08-26)
  *) echo "❌ 未知 VARIANT=$VARIANT (可选 xy_rb / xy / redbox / xy_rb_mb)"; exit 1 ;;
esac

if [ "$STEPS" -le 5 ]; then
  TAG="_smoke_$VARIANT"; SAVE_FREQ="${SAVE_FREQ:--1}"
else
  TAG="${TAG:-vopd-$VARIANT}"; SAVE_FREQ="${SAVE_FREQ:-20}"
fi
CKPT="${CKPT:-$VOPD_ROOT/checkpoints/$TAG}"
ROLLOUT="${ROLLOUT:-$VOPD_ROOT/rollouts/$TAG}"

fail() { echo "❌ $*"; exit 1; }

# ------------------------------------------------------------------ 前置检查
[ -f "$TRAIN_FILE" ] || fail "缺训练数据: $TRAIN_FILE  (先跑 25_prep_data.py --variant $VARIANT)"

n=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || true)
[ "${n:-0}" -ge 2 ] || fail "只看到 ${n:-0} 张卡, 配置按 2 张调的"

busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F', ' '$2>2000{print "  GPU "$1": "$2" MiB"}')
if [ -n "$busy" ]; then
  echo "❌ GPU 上还有东西占着显存:"; echo "$busy"
  echo "   多半是探针/评测用的 vLLM。停掉:  PORT=8766 bash $SETUP/06_serve.sh stop"
  echo "   若仍有残留(子进程 VLLM::EngineCore, pkill 匹配不到)，nvidia-smi 查 PID 后 kill -9"
  exit 1
fi

# flash-attn 不带 PTX, 换卡必须重编。
# ⚠️ 曾经这里用 `.flash_attn_archs`(编译时写下的) 去比 .so 里的 arch —— 两边来自同一次
#    编译, 是自证循环, **永远为真**, 跟当前用的是哪张卡毫无关系。真调度到别的卡照样放行,
#    然后运行时炸 "no kernel image is available"。
#    正确做法: want 取 00_env.sh 从 nvidia-smi 实时推出来的 $FLASH_ATTN_CUDA_ARCHS。
want="${FLASH_ATTN_CUDA_ARCHS:-}"
[ -n "$want" ] || fail "00_env.sh 没能从 nvidia-smi 推出架构(FLASH_ATTN_CUDA_ARCHS 为空), 拒绝盲跑"
have=$(cuobjdump --list-elf "$VOPD_ENV"/lib/python3.12/site-packages/flash_attn_2_cuda*.so 2>/dev/null \
       | grep -oE "sm_[0-9]+" | sort -u | tr '\n' ' ')
[ -n "$have" ] || fail "读不出 flash-attn 的 arch(cuobjdump 失败或 .so 不存在)"
case " $have " in
  *" sm_$want "*) echo ">>> flash-attn: [$have] 含当前卡 sm_$want ✓" ;;
  *) fail "flash-attn 编译为 [$have], 当前卡是 sm_$want。重编: SKIP=0 bash 03_build_ext.sh" ;;
esac

# teacher 走哪条通路完全由 parquet 的列决定, 走错了不会报错、只会静默变成别的实验:
#   有 teacher_prompt  -> 文本模板通路(ray_trainer.py:1300 -> 795 -> 761), 模板里的
#                         <image> 由 bbox_images 填充, 所以**文本和图像特权同时生效**
#   没有               -> 仅图像替换(_swap_images_in_messages)
WANT_TP="$WANT_TP" WANT_DIFF="$WANT_DIFF" "$VOPD_PY" - "$TRAIN_FILE" <<'PY' || exit 1
import os, re, sys, pyarrow.parquet as pq
t = pq.read_table(sys.argv[1]); cols = t.schema.names
want_tp   = os.environ["WANT_TP"] == "1"
want_diff = os.environ["WANT_DIFF"] == "1"
has_tp    = "teacher_prompt" in cols
if "bbox_images" not in cols: sys.exit(f"❌ 缺 bbox_images 列; 实际: {cols}")
if want_tp and not has_tp: sys.exit("❌ 该变体需要 teacher_prompt 列, 但没有 —— 会静默退化成纯图像特权")
if not want_tp and has_tp: sys.exit("❌ 该变体不该有 teacher_prompt 列, 但有 —— 会额外引入文本特权")

# 全量扫, 不抽样 —— 6241 条纯字符串比较 < 1s。抽查前 500 条的话, 万一以后 prep 脚本
# 改成按条件写不同 prompt, 前 500 过了不代表全量过。
d = t.to_pylist(); N = len(d)
bad = []

same = sum(1 for r in d if r["images"][0]["path"] == r["bbox_images"][0]["path"])
# 视觉特权的存在性必须**按变体双向检查**。只查一边的话, 另一边静默失效时
# 实验会悄悄变成另一个实验(xy_rb -> 纯语言), 而运行期指标发现不了。
if want_diff and same:
    bad.append(f"{same}/{N} 条 student/teacher 图相同, 但该变体要求不同 -> 视觉特权失效")
if not want_diff and same != N:
    bad.append(f"{N-same}/{N} 条 student/teacher 图不同, 但该变体要求相同 -> 混进了视觉特权")

if has_tp:
    n = sum(1 for r in d
            if r["prompt"][0]["content"] !=
               [m for m in r["teacher_prompt"] if m["role"] == "user"][0]["content"])
    if n: bad.append(f"{n}/{N} 条 teacher/student 的 user 轮不一致 -> 腔调会分岔")
    # teacher_prompt 结构: 必须是 (system, user) 两条且 content 为 str。
    # content 是 list 的话 _build_teacher_messages_from_template(ray_trainer.py:764)
    # 会 continue 跳过, image_offset 停在 0, 最后在 :787 报错 —— 但那要等模型加载完几分钟。
    n = sum(1 for r in d
            if [m["role"] for m in r["teacher_prompt"]] != ["system", "user"]
            or not all(isinstance(m["content"], str) for m in r["teacher_prompt"]))
    if n: bad.append(f"{n}/{N} 条 teacher_prompt 结构不是 (system:str, user:str)")
    # system 里必须真的含全部四个坐标(防 format 失败留下 '{}' 或坐标全 0)
    n = sum(1 for r in d
            if not all(str(v) in set(re.findall(r"\d+", r["teacher_prompt"][0]["content"]))
                       for v in r["extra_info"]["bbox_norm1000"]))
    if n: bad.append(f"{n}/{N} 条 teacher system 里没有完整坐标 -> 语言特权是空的")

# <image> 占位符数必须等于图片数。不等时 ray_trainer.py:770/787 会 raise, 但那是
# 模型加载几分钟之后才炸, 不如现在就拦。
n = sum(1 for r in d if r["prompt"][0]["content"].count("<image>") != len(r["images"]))
if n: bad.append(f"{n}/{N} 条 student 的 <image> 数 != images 数")
if has_tp:
    n = sum(1 for r in d
            if sum(m["content"].count("<image>") for m in r["teacher_prompt"]) != len(r["bbox_images"]))
    if n: bad.append(f"{n}/{N} 条 teacher 的 <image> 数 != bbox_images 数")

# student 的 prompt 里绝不能出现完整坐标(四个值同时作为独立数字 token 出现)
n = sum(1 for r in d
        if all(str(v) in set(re.findall(r"\d+", r["prompt"][0]["content"]))
               for v in r["extra_info"]["bbox_norm1000"]))
if n: bad.append(f"{n}/{N} 条 student prompt 里泄漏了完整坐标")

if bad:
    for b in bad: print("❌ " + b, file=sys.stderr)
    sys.exit(1)
print(f"    通路检查 ✓  {N} 条 | 语言特权={has_tp} | student/teacher 图"
      f"{'不同' if want_diff else '相同(该变体如此)'} | 坐标零泄漏")
PY

# Ray 的 AF_UNIX socket 路径上限 107 字节, Ray 自己的后缀固定占 67 -> 这里最多 40。
# $TMPDIR/ray_<jobid> 是 41 字节, 超 1 个字节就起不来(踩过)。用定长短名。
export RAY_TMPDIR="$TMPDIR/ray_t"
[ ${#RAY_TMPDIR} -le 40 ] || fail "RAY_TMPDIR 过长(${#RAY_TMPDIR}>40)"
export RAY_memory_monitor_refresh_ms=0   # 真正杀 WorkerDict 的是 raylet 的 C++ monitor
export WORLD_SIZE=1                      # run_vision_opd.sh:41 在 set -u 下用它
mkdir -p "$RAY_TMPDIR" "$CKPT" "$ROLLOUT"

RESUMING=0
if [ -f "$CKPT/latest_checkpointed_iteration.txt" ]; then
  if [ "$STEPS" -le 5 ]; then
    echo ">>> 冒烟: 清掉旧 checkpoint 从头开始"; rm -rf "$CKPT"; mkdir -p "$CKPT"
  else
    RESUMING=1
    echo ">>> ⚠️  已有 checkpoint (step $(cat "$CKPT/latest_checkpointed_iteration.txt")), resume_mode=auto 会续训"
    echo ">>> ⚠️  注意: **EMA teacher 不在 checkpoint 里**。FSDPCheckpointManager"
    echo "        (fsdp_workers.py:935) 只存 actor_module_fsdp, ref/teacher 既不存也不载。"
    echo "        teacher_update_rate=0.05 => 有效时间常数约 20 步, 续训等于把 teacher"
    echo "        重置回底模再追赶。**本轮 120 步应当一次跑完**(约 8.5h, 上限 24h)。"
  fi
fi
# rollout dump 的文件名是 f"{global_steps}.jsonl"(ray_trainer.py:500), 重跑只覆盖不删。
# 上次跑到 120 步、这次只跑 60 步的话, 61~120.jsonl 是**上个实验的**, 而
# 28_analyze_rollouts.py 是 glob 整个目录 -> 分析结果会混入旧实验。
if [ "$RESUMING" = "0" ] && [ -n "$(ls -A "$ROLLOUT" 2>/dev/null)" ]; then
  echo ">>> 清掉 $ROLLOUT 里的旧 rollout dump ($(ls "$ROLLOUT" | wc -l) 个文件)"
  rm -f "$ROLLOUT"/*.jsonl
fi

echo "=========================================================="
echo " 变体 : $VARIANT"
echo " GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ×$n"
echo " 步数 : $STEPS   存档间隔: $SAVE_FREQ"
echo " 数据 : $TRAIN_FILE"
echo " ckpt : $CKPT"
echo "=========================================================="
echo

cd "$VOPD_ROOT"
# 相对论文原设置的偏离及理由(与前几轮一致, 一次只改一个变量):
#   train_batch_size 96->24 / n_gpus 8->2   资源限制
#   三个 offload True->False                论文为 8×80GB 设计; 2×96GB 不需要,
#                                           实测开着慢 3.5 倍。**不改变梯度**
#   gpu_memory_utilization 0.7->0.45        只影响 vLLM KV cache, 不改变采样结果
#   lr_warmup_steps 10->40                  按 warmup 期间看到的数据量对齐论文
#                                           (论文 10×768 = 我们 40×192)
#   lr 保持 2e-6 不变
#   **不设 max_actor_ckpt_to_keep** —— 上一轮设了 2, 把 step 50/100/150/200 全删了,
#   最有价值的那个(IoU 0.776 + 答题率 95.3%)没保住。磁盘 12T, 不差这点。
bash scripts/run_vision_opd.sh \
    data.train_files="[\"$TRAIN_FILE\"]" \
    data.train_batch_size=24 \
    data.dataloader_num_workers=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps=40 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.logger='["console"]' \
    trainer.total_training_steps="$STEPS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.experiment_name="$TAG" \
    trainer.default_local_dir="$CKPT" \
    trainer.rollout_data_dir="$ROLLOUT" \
    "$@"
rc=$?

echo
echo "=========================================================="
echo " 结束  exit=$rc   $(date)"
if [ "$rc" -ne 0 ]; then
  echo " 关键报错:"
  echo " ckpt: $(ls -1 "$CKPT" 2>/dev/null | tr '\n' ' ')"
  exit $rc
fi
# 这次要看的三件事, 与前两版的失败模式一一对应:
#   ① teacher 通路      swap_fraction 必须 1.0, JSD 不能≈0
#   ② 答题率            v2 从 97.4% 掉到 16.1%, 这次必须保持住
#   ③ student bbox IoU  探针里学生基线 0.368, teacher 0.803, 应该往上走
"$VOPD_PY" "$SETUP/28_analyze_rollouts.py" "$ROLLOUT" "$TRAIN_FILE" \
  || echo " (分析失败, 手动跑: \$VOPD_PY 28_analyze_rollouts.py $ROLLOUT $TRAIN_FILE)"
echo "=========================================================="
exit $rc
