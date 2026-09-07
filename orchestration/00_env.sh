#!/bin/bash
# Vision-OPD 环境变量 / module —— 每次开新 shell 都 source 这个文件
# 用法: source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh

SCRATCH=/sfs/weka/scratch/nkw3mr
export VOPD_ROOT=$SCRATCH/Vision-OPD
export VOPD_ENV=$SCRATCH/envs/vision-opd

# --- modules ---
module purge
module load miniforge/26.3.2
module load cuda/12.8.0                 # 必须 12.8: torch 2.10.0 是 cu128 构建, B200=sm_100
module load cudnn/9.8.0-CUDA-12.8.0
module load gcc/11.4.0                  # 系统 gcc 8.5 太旧, 编不了 torch 2.10 的 C++17 扩展

export CUDA_HOME=${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}

# --- 所有缓存挪到 scratch, 别写爆 home 配额 ---
export CONDA_ENVS_PATH=$SCRATCH/envs
export CONDA_PKGS_DIRS=$SCRATCH/.cache/conda_pkgs
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
export HF_HOME=$SCRATCH/.cache/huggingface
export TORCH_HOME=$SCRATCH/.cache/torch
export TRITON_CACHE_DIR=$SCRATCH/.cache/triton
export VLLM_CACHE_ROOT=$SCRATCH/.cache/vllm
export XDG_CACHE_HOME=$SCRATCH/.cache
export TMPDIR=$SCRATCH/tmp
# FlashInfer 会在运行时 JIT 出 sm_100 kernel, 缓存默认落在 ~/.cache/flashinfer (走 expanduser,
# 不吃 XDG_CACHE_HOME), 必须显式指到 scratch, 否则写爆 home 配额。
export FLASHINFER_WORKSPACE_BASE=$SCRATCH/.cache/flashinfer
# vllm 的 usage 上报线程要写 ~/.config/vllm/, 在这台机器上路径解析成了 /.config (只读) ->
# 后台线程抛 OSError(Errno 30)。主流程不受影响, 但日志很吵。顺带关掉遥测上报
# (它默认会把使用统计发到外部服务器, HPC 上没必要)。
export VLLM_CONFIG_ROOT=$SCRATCH/.cache/vllm/config
export XDG_CONFIG_HOME=$SCRATCH/.config
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$VLLM_CONFIG_ROOT" "$XDG_CONFIG_HOME" \
         "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE" "$TMPDIR"

# --- 只为当前这块卡编译, 省掉大量编译时间 ---
# 按可见 GPU 自动推导, 探测不到(登录节点/无 GPU)就退回 10.0 (B200)。
#   B200          -> 10.0 (sm_100)
#   RTX PRO 6000  -> 12.0 (sm_120)   Blackwell 工作站卡, 和 B200 不同族, cubin 不通用
#   A6000 / A40   ->  8.6 (sm_86)
# 可以手动覆盖: VOPD_ARCH=12.0 source 00_env.sh
_vopd_cc="${VOPD_ARCH:-}"
if [ -z "$_vopd_cc" ] && command -v nvidia-smi >/dev/null 2>&1; then
  # 末尾的 `|| true` 不能省: 本文件会被 `set -euo pipefail` 的脚本 source
  # (09_train_smoke.sh:13)。无驱动的机器上 nvidia-smi 退出码是 9, 而 pipefail
  # 会把它传给整条管道, 于是这个赋值失败 -> 调用方在 source 这一行直接静默退出,
  # 连一个字都不打印。而且那样的话下面的格式校验根本轮不到执行。
  _vopd_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
             | tr -d ' ' | sort -u | paste -sd';' - || true)
  # nvidia-smi 在没有驱动的机器上(比如登录节点)会把报错打到 **stdout**, 于是那段
  # "NVIDIA-SMI has failed because..." 会被当成 arch 塞进环境变量。必须校验格式:
  # 只接受 "12.0" 或 "10.0;12.0" 这种。
  case "$_vopd_cc" in
    *[!0-9.\;]* | "" ) _vopd_cc="" ;;
  esac
fi
export TORCH_CUDA_ARCH_LIST="${_vopd_cc:-10.0}"
# flash-attn **不读** TORCH_CUDA_ARCH_LIST, 只认 FLASH_ATTN_CUDA_ARCHS,
# 且格式是去掉小数点的 "100"/"120"/"86"。这里一并推导, 免得换卡时漏掉。
export FLASH_ATTN_CUDA_ARCHS="$(echo "$TORCH_CUDA_ARCH_LIST" | tr -d '.' )"

# --- 编译缓存按 arch 分开 ---
# 2026-08-22 踩的坑: 从 B200(sm_100) 换到 RTX PRO 6000(sm_120) 后, Inductor/Triton
# 把昨天为 sm_100 编的 cubin 喂给了 sm_120 的卡, 报
#     torch._inductor.exc.InductorError: RuntimeError: CUDA driver error: file not found
# (即 cuModuleLoadData 的 CUDA_ERROR_FILE_NOT_FOUND=301)。它的缓存键没能区分这两个
# 架构。所以这里把路径带上 arch 后缀, 各编各的, 换卡不用手动清缓存。
_vopd_tag="sm$(echo "$FLASH_ATTN_CUDA_ARCHS" | tr ';' '_')"
export TRITON_CACHE_DIR="$SCRATCH/.cache/triton-$_vopd_tag"
export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm-$_vopd_tag"
export VLLM_CONFIG_ROOT="$VLLM_CACHE_ROOT/config"
export FLASHINFER_WORKSPACE_BASE="$SCRATCH/.cache/flashinfer-$_vopd_tag"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.cache/inductor-$_vopd_tag"
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" \
         "$FLASHINFER_WORKSPACE_BASE" "$TORCHINDUCTOR_CACHE_DIR"
unset _vopd_cc _vopd_tag

# --- B200 注意力后端 ---
# vllm 0.18.0 预编译产物里 _vllm_fa3_C 只有 sm_90a 且不带 PTX -> FA3 在 B200 上不可用;
# _vllm_fa2_C 只有 sm_80 cubin, 靠 PTX JIT 上 sm_100, 能跑但用不上 Blackwell 指令。
# Blackwell 的正路是 FlashInfer (运行时 nvcc JIT 出 sm_100 原生 kernel)。
# 因此 cuda module 在"运行时"也必须是 loaded 的 —— 上面已 load, 别在跑任务时 module purge。
#
# 不主动 export VLLM_ATTENTION_BACKEND: vllm 会按 device capability 自动选后端,
# 而且 run_vision_opd.sh 里明确写了 `unset VLLM_ATTENTION_BACKEND`, 强行设会跟它打架。
# 只有在自动选择出问题时才手动打开下面这行:
# export VLLM_ATTENTION_BACKEND=FLASHINFER

# 激活 env (如果已创建)
if [ -d "$VOPD_ENV" ]; then
  conda activate "$VOPD_ENV"

  # !! 必须放在 conda activate 之后 !!
  # `module load gcc/11.4.0` 会把 gcc 自己的 lib64 插到 LD_LIBRARY_PATH 前面, 于是运行时
  # 用的是 gcc 11.4 的 libstdc++.so.6 —— 它只提供到 CXXABI_1.3.13。而 env 里的
  # libicui18n.so.78 等库需要 CXXABI_1.3.15, 结果 `import sqlite3` 就炸:
  #     ImportError: ... libstdc++.so.6: version `CXXABI_1.3.15' not found
  # conda env 自带的 libstdc++.so.6.0.35 提供到 1.3.17, 让它优先即可。
  # (libstdc++ 向后兼容: 1.3.17 能满足所有对更低版本的需求, 所以对用 gcc 11.4 编出来的
  #  flash-attn 等扩展也没有影响。)
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

  # 不要依赖 PATH 去找解释器。`module purge` + 继承来的 conda 状态 + 嵌套 activate
  # 有可能让 `python` 落回系统里的 anaconda 3.11 (它在基础 PATH 里写死), 那个 env 里
  # 没有我们装的包, 症状是莫名其妙的 ModuleNotFoundError。
  # 脚本里一律用 $VOPD_PY, 交互式用 python 也行(下面把 env/bin 强制前置了)。
  export VOPD_PY="$VOPD_ENV/bin/python"
  export PATH="$VOPD_ENV/bin:$PATH"
fi
