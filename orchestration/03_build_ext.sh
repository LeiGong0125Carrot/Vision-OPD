#!/bin/bash
# 编译 flash-attn 和 causal-conv1d。这是整个流程最慢、最容易失败的一步。
# 建议 nohup 后台跑, 别占着交互 shell。
set -euo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
conda activate "$VOPD_ENV"

# 每个编译 job 峰值约 6-8GB 内存。跟着 Slurm 实际分到的 CPU 走, 设多了只会互相抢 + OOM。
# 本 job: cpu=8, mem=128G -> MAX_JOBS=8 (8x8=64G, 安全)
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export TORCH_CUDA_ARCH_LIST="10.0"     # 给 causal-conv1d 等走 torch 扩展惯例的包用
export FLASH_ATTENTION_FORCE_BUILD="TRUE"

# !! 关键: flash-attn 2.8.x 不读 TORCH_CUDA_ARCH_LIST, 它有自己的 FLASH_ATTN_CUDA_ARCHS,
#    默认值是 "80;90;100;120" —— 也就是说不设的话会把 4 个 arch 全编一遍, 时间 x4。
#    B200 = sm_100, 只编这一个。(RTX PRO 6000 改成 120; H200 改成 90)
#    flash-attn 官方支持 sm_100/sm_120, 但要求 CUDA >= 12.8 —— 我们正好是 12.8。
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-100}"

echo ">>> MAX_JOBS=$MAX_JOBS  TORCH_ARCH=$TORCH_CUDA_ARCH_LIST  FLASH_ATTN_ARCHS=$FLASH_ATTN_CUDA_ARCHS  ($(date))"

# 断点续跑: 已经编好的就跳过。pip 不缓存编译产物, 重编一次是 30 分钟起,
# 所以 job 被杀后重提 sbatch 时这个判断很值钱。SKIP=0 可强制重编。
have() { python -c "import $1" 2>/dev/null; }

echo ">>> [1/2] flash-attn  (73 个 .cu, 单 arch 预计 30min; 不设 FLASH_ATTN_CUDA_ARCHS 会是 4 倍)"
if [ "${SKIP:-1}" = "1" ] && have flash_attn; then
  echo "    已安装 $(python -c 'import flash_attn;print(flash_attn.__version__)'), 跳过。SKIP=0 可强制重编。"
else
  # !! 必须 --no-cache-dir: pip 会缓存我们自编的 wheel, 但 wheel 文件名里不含 arch 信息。
  #    我们用 code=sm_XXX 编译(连 PTX 都没有), 所以缓存里那个 wheel 只认那一种卡。
  #    换卡后若命中缓存, 运行时会报 "no kernel image is available for execution on
  #    the device", 且完全不会提示是缓存导致的。重编时一律绕开缓存。
  #
  # !! 还必须先 uninstall: 2026-08-22 踩过 —— SKIP=0 绕过了上面的跳过逻辑, 但
  #    `pip install flash-attn` 看到已安装的 2.8.3.post1 就直接 "Requirement already
  #    satisfied" 什么也没干, 于是从 B200 换到 RTX PRO 6000 后 .so 里还是只有 sm_100,
  #    训练跑到 flash_attn_gpu.varlen_fwd 才炸。--no-cache-dir 只管 wheel 缓存,
  #    **不会**强制重装已满足的依赖。
  #    (不能只加 --force-reinstall: 那会连 torch 一起重装。必须配 --no-deps。)
  echo "    先卸载旧的 flash-attn (避免 pip 判定 already satisfied 而跳过重编)"
  pip uninstall -y flash-attn 2>/dev/null || true
  pip install flash-attn --no-build-isolation --no-cache-dir --no-deps -v
fi

# 记录这次编译的 arch 并**核对实际产物** —— 标记文件曾经和产物对不上, 反而误导人。
echo "$FLASH_ATTN_CUDA_ARCHS" > "$VOPD_ENV/.flash_attn_archs"
_fa_so=$(ls "$VOPD_ENV"/lib/python3.12/site-packages/flash_attn_2_cuda*.so 2>/dev/null | head -1)
if [ -n "$_fa_so" ] && command -v cuobjdump >/dev/null 2>&1; then
  _fa_sass=$(cuobjdump --list-elf "$_fa_so" 2>/dev/null | grep -oE "sm_[0-9]+[a-z]*" | sort -u | tr '\n' ' ')
  echo "    flash-attn 实际编进的 SASS: ${_fa_sass:-<无>}   (期望含 sm_$FLASH_ATTN_CUDA_ARCHS)"
  case "$_fa_sass" in
    *"sm_$FLASH_ATTN_CUDA_ARCHS"*) echo "    ✅ 匹配" ;;
    *) echo "    ❌ 不匹配! 当前这张卡跑起来会报 'no kernel image is available for execution on the device'"
       echo "       用 SKIP=0 重跑本脚本。"; exit 1 ;;
  esac
fi

echo ">>> [2/2] causal-conv1d==1.6.1  (3 个 .cu x 9 个 arch, 预计 5-15min)"
if [ "${SKIP:-1}" = "1" ] && have causal_conv1d; then
  echo "    已安装, 跳过。SKIP=0 可强制重编。"
else
  pip install causal-conv1d==1.6.1 --no-build-isolation -v
fi

echo ">>> 编译完成 ($(date))"
python -c "import flash_attn, causal_conv1d; print('flash_attn', flash_attn.__version__); print('causal_conv1d ok')"
