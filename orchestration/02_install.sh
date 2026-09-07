#!/bin/bash
# 建 conda env + 装纯 wheel 依赖。不含需要编译的 flash-attn / causal-conv1d。
set -euo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh

if [ ! -d "$VOPD_ENV" ]; then
  echo ">>> 创建 python 3.12 环境: $VOPD_ENV"
  conda create -p "$VOPD_ENV" python=3.12 -y
fi
conda activate "$VOPD_ENV"
python -V

echo ">>> 升级 pip"
pip install --upgrade pip

# vllm 单独处理: 它是 258 个 pin 里唯一与本机 glibc 2.28 不兼容的包 (只发 manylinux_2_31 wheel)。
# 留在 requirements 里会让整条 pip install 直接失败, 所以先剔除, 交给 02b_fix_vllm.sh。
REQ="$TMPDIR/requirements-novllm.txt"
grep -v '^vllm==' "$VOPD_ROOT/requirements.txt" > "$REQ"
echo ">>> 已从 requirements 剔除 vllm ($(wc -l < "$REQ") 个包待装)"

echo ">>> 安装 requirements (--no-deps, 约 15-25GB 下载)"
# --no-deps 是 README 的要求: 版本已全部 pin 死, 让 pip 解析反而会打架。
# torch 2.10.0 的 PyPI 默认 wheel 就是 cu128, 与 requirements 里的 nvidia-*==12.8.x pin 一致,
# 所以这里不需要 --index-url download.pytorch.org。
# 其中 antlr4-python3-runtime / gpustat / pylatexenc / word2number 只有 sdist,
# 但都是纯 Python, 秒装, 不需要编译器。
pip install --no-deps -r "$REQ"

echo ">>> 以 editable 方式安装 verl 本体"
pip install -e "$VOPD_ROOT" --no-deps

echo ">>> 校验 torch 能看到 B200"
python - <<'PY'
import torch
print("torch     :", torch.__version__)
print("built cuda:", torch.version.cuda, "  (期望 12.8)")
print("available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0  :", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), "(期望 (10,0))")
    print("arch list :", torch.cuda.get_arch_list())
PY
echo
echo ">>> 02 完成。下一步: bash 02b_fix_vllm.sh"
