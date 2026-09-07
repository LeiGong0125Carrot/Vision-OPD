#!/bin/bash
# 单独处理 vllm —— 唯一一个和本机 glibc 不兼容的包。
#
# 问题: 节点是 Rocky 8.10 (glibc 2.28), 而 vllm 0.16~0.18 全系列只发 manylinux_2_31 wheel。
#       pip 认为没有可用 wheel -> 回退 sdist -> sdist metadata 是 0.18.0+cu128 与请求的
#       0.18.0 不符 -> "No matching distribution found for vllm==0.18.0"。
#
# 实测结论: wheel 里 7 个 .so 中 6 个只需 GLIBC<=2.14, 唯一超标的是 _moe_C.abi3.so 的
#       一个符号 log2@GLIBC_2.29。manylinux_2_31 这个 tag 是打包时声明的, 远高于真实需求。
#
# 修复三步:
#   1) 把 wheel 的 platform tag 从 manylinux_2_31 改成 manylinux_2_28 (只需改文件名, pip 按文件名判兼容)
#   2) patchelf --clear-symbol-version log2   -> 让符号变成无版本引用
#   3) fix_glibc_verneed.py                   -> 把 .gnu.version_r 里对 libm 的 GLIBC_2.29
#                                                需求重指向 .dynstr 中已存在的 GLIBC_2.2.5
#      (log2@GLIBC_2.2.5 与 2.29 版数值结果一致, 差别只在 SVID errno/matherr 边界行为)
set -euo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
conda activate "$VOPD_ENV"

SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
WHLDIR=/sfs/weka/scratch/nkw3mr/tmp
SRC_WHL="$WHLDIR/vllm-0.18.0-cp38-abi3-manylinux_2_31_x86_64.whl"
DST_WHL="$WHLDIR/vllm-0.18.0-cp38-abi3-manylinux_2_28_x86_64.whl"

mkdir -p "$WHLDIR"

# --- 1) 拿到 wheel ---
if [ ! -f "$SRC_WHL" ]; then
  echo ">>> 下载 vllm 0.18.0 wheel (414MB)"
  python - <<'PY'
import urllib.request, json
d = json.load(urllib.request.urlopen('https://pypi.org/pypi/vllm/0.18.0/json', timeout=60))
fn = 'vllm-0.18.0-cp38-abi3-manylinux_2_31_x86_64.whl'
url = [f['url'] for f in d['urls'] if f['filename'] == fn][0]
urllib.request.urlretrieve(url, '/sfs/weka/scratch/nkw3mr/tmp/' + fn)
print('downloaded')
PY
fi

# --- 1b) 改 platform tag (硬链接, 不占额外空间) ---
if [ ! -f "$DST_WHL" ]; then
  ln -f "$SRC_WHL" "$DST_WHL"
fi
echo ">>> wheel: $(basename "$DST_WHL")"

# --- 2) 安装 ---
echo ">>> 安装 patchelf + vllm"
pip install --no-deps --quiet patchelf
# 重装会覆盖掉已打的补丁, 所以已装同版本就跳过。要强制重装: FORCE=1 bash 02b_fix_vllm.sh
if [ "${FORCE:-0}" = "1" ] || ! python -c "import vllm" 2>/dev/null; then
  pip install --no-deps "$DST_WHL"
else
  echo "    vllm 已安装且可 import, 跳过重装 (补丁保留)。FORCE=1 可强制重装。"
fi

SP=$(python -c "import site; print(site.getsitepackages()[0])")
echo ">>> site-packages: $SP"

# --- 3) 打补丁 ---
echo ">>> 修补 vllm 的 .so"
python "$SETUP/fix_glibc_verneed.py" --dry-run "$SP/vllm/_moe_C.abi3.so" >/dev/null 2>&1 || true
for so in $(find "$SP/vllm" -name "*.so"); do
  # 先清掉所有 glibc>2.28 的符号版本引用
  for sym in $(readelf --dyn-syms --wide "$so" 2>/dev/null \
                 | grep -oE "[A-Za-z0-9_]+@GLIBC_2\.(29|3[0-9]|[4-9][0-9])" \
                 | cut -d@ -f1 | sort -u); do
    echo "    patchelf --clear-symbol-version $sym  $(basename "$so")"
    patchelf --clear-symbol-version "$sym" "$so"
  done
  # 再修 verneed
  python "$SETUP/fix_glibc_verneed.py" "$so" 2>/dev/null | grep -v "无需处理" || true
done

# --- 4) 验证 ---
# 交给 02c (它不带 set -e, 会把所有检查跑完再报告)。
echo
exec bash "$SETUP/02c_verify_vllm.sh"
