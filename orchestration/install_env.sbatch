#!/bin/bash
#SBATCH --job-name=vopd-install
#SBATCH --account=zhangmlgroup
#SBATCH --partition=standard          # 纯 CPU 分区: 7 天时限, 301 节点, 不占 GPU 配额
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # 编译并行度; 16x8GB=128GB 峰值内存
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/install-%j.out
#SBATCH --error=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/install-%j.err
#
# 一次性把整个环境装完 —— 全程不需要 GPU。
#
# 为什么能在 CPU 分区跑:
#   - pip 下载/安装: 不用 GPU
#   - flash-attn / causal-conv1d 编译: 只需要 nvcc。我们显式设了 FLASH_ATTN_CUDA_ARCHS
#     和 TORCH_CUDA_ARCH_LIST, 所以 setup.py 不会去查设备能力 -> 无 GPU 也能编。
#   - 唯一需要 GPU 的是最后的 kernel 实跑验证, 02c 检测到无 GPU 会自动跳过。
#
# 提交:  sbatch install_env.sbatch
# 跟进:  tail -f logs/install-<jobid>.out
# 装完后在 GPU 节点上跑: bash 04_verify.sh
set -euo pipefail

SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
mkdir -p "$SETUP/logs"
cd "$SETUP"

echo "=============================================="
echo "节点   : $(hostname)"
echo "CPU    : ${SLURM_CPUS_PER_TASK:-?}   内存: ${SLURM_MEM_PER_NODE:-?}MB"
echo "开始   : $(date)"
echo "=============================================="

# MAX_JOBS 跟着 Slurm 给的 CPU 走 (00_env.sh / 03 会读 SLURM_CPUS_PER_TASK)
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"

t0=$SECONDS
echo; echo "########## [1/3] 依赖安装 (02) ##########"
bash 02_install.sh
echo "--- 用时 $(( (SECONDS-t0)/60 )) 分钟 ---"

t1=$SECONDS
echo; echo "########## [2/3] vllm + glibc 补丁 (02b) ##########"
bash 02b_fix_vllm.sh
echo "--- 用时 $(( (SECONDS-t1)/60 )) 分钟 ---"

t2=$SECONDS
echo; echo "########## [3/3] 编译扩展 (03) ##########"
bash 03_build_ext.sh
echo "--- 用时 $(( (SECONDS-t2)/60 )) 分钟 ---"

echo
echo "=============================================="
echo "全部完成, 总用时 $(( (SECONDS-t0)/60 )) 分钟   $(date)"
echo "下一步: 在 GPU 节点上跑  bash 04_verify.sh"
echo "=============================================="
