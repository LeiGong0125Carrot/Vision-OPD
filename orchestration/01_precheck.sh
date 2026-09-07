#!/bin/bash
# 在 B200 计算节点上跑。确认硬件 / 工具链 / 外网。任何一项 FAIL 都先别往下走。
set -u
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh

echo "===== [1] GPU ====="
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv || echo "FAIL: 没有 GPU"
echo "  期望: 8x B200, compute_cap 10.0, driver >= 570 (cu128 要求)"

echo; echo "===== [2] 工具链 ====="
echo "nvcc : $(nvcc --version 2>/dev/null | tail -1)"
echo "gcc  : $(gcc --version | head -1)     # 需要 >= 11"
echo "conda: $(conda --version)"
echo "CUDA_HOME=$CUDA_HOME"

echo; echo "===== [3] 计算节点外网 (最关键的未知项) ====="
timeout 30 python3 -c "
import urllib.request
for url in ['https://pypi.org/simple/torch/','https://huggingface.co']:
    try:
        urllib.request.urlopen(url,timeout=15); print(f'  OK   {url}')
    except Exception as e: print(f'  FAIL {url}  -> {type(e).__name__}: {e}')
"
echo "  若 FAIL: 需改在 login node 上做 pip 安装(见计划 Plan B)"

echo; echo "===== [4] 资源 ====="
echo "CPU核数: $(nproc)   内存: $(free -g | awk '/^Mem/{print $2}')GB"
df -h /sfs/weka/scratch/nkw3mr | tail -1
echo "  编译 flash-attn 需要 >= 32GB 内存空闲; 环境+缓存约需 120GB 磁盘"
