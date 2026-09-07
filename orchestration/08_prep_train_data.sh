#!/bin/bash
# 准备训练数据 (Vision-OPD-6K)。
#
# 这个脚本自己做下载和解包, 然后用 prepare_data.py --skip-download 只做 parquet 转换。
# 之所以不直接用 prepare_data.py 全流程, 是因为它有两个在当前环境下会挂的地方:
#
#   1) 第 39 行调 `huggingface-cli`, 但 huggingface_hub 1.9.0 已把 CLI 改名成 `hf`
#      -> FileNotFoundError。装个转发 shim 解决。
#   2) 第 48 行 `cat images.tar.gz0* | tar -xf - -C .` 解学生图像。分卷拼起来是 gzip
#      压缩的 tar, 而 GNU tar 1.30 从**管道**读时不会自动识别压缩, 报:
#          tar: Archive is compressed. Use -z option
#      (它解 teacher_images.tar.gz 用的是 `tar -xf <文件>`, 对文件能自动识别, 所以没事。)
#
#   注意: 曾试过"自己解包 + 删掉分卷, 让 prepare_data.py 发现没文件而跳过" —— 行不通,
#   因为它的 download_dataset() 会无条件重新下载, 把删掉的分卷补回来再撞同一个 bug。
#   正解是用它自带的 --skip-download。
set -euo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh

DATA_DIR="${DATA_DIR:-$VOPD_ROOT/data}"
REPO="yuanqianhao/Vision-OPD-6K"

# --- 1) huggingface-cli -> hf 转发 shim ---
SHIM="$VOPD_ENV/bin/huggingface-cli"
if [ ! -x "$SHIM" ]; then
  echo ">>> 安装 huggingface-cli -> hf 转发 shim"
  printf '#!/bin/bash\nexec "%s/bin/hf" "$@"\n' "$VOPD_ENV" > "$SHIM"; chmod +x "$SHIM"
fi

# --- 2) 下载 (幂等, 已有的会跳过) ---
echo ">>> 下载 $REPO -> $DATA_DIR"
"$VOPD_ENV/bin/hf" download --repo-type dataset "$REPO" --local-dir "$DATA_DIR"

# --- 3) 解包学生图像 (用 -z) ---
IMG_DIR="$DATA_DIR/images"
if compgen -G "$IMG_DIR/images.tar.gz0*" > /dev/null; then
  echo ">>> 解包学生图像 ($(ls "$IMG_DIR"/images.tar.gz0* | wc -l) 个分卷, 约 28GB, 数分钟)"
  ( cd "$IMG_DIR" && cat $(ls images.tar.gz0* | sort) | tar -xzf - -C . )
  rm -f "$IMG_DIR"/images.tar.gz0*        # set -e: 解包失败不会走到这
  echo "    完成, 共 $(find "$IMG_DIR" -type f | wc -l) 个文件"
else
  echo ">>> 学生图像已解包 ($(find "$IMG_DIR" -type f 2>/dev/null | wc -l) 个文件)"
fi

# --- 4) 解包 teacher 图像 ---
T_DIR="$DATA_DIR/teacher_images"
if [ -f "$T_DIR/teacher_images.tar.gz" ]; then
  echo ">>> 解包 teacher 图像"
  ( cd "$T_DIR" && tar -xzf teacher_images.tar.gz -C . )
  rm -f "$T_DIR/teacher_images.tar.gz"
  echo "    完成, 共 $(find "$T_DIR" -type f | wc -l) 个文件"
else
  echo ">>> teacher 图像已解包 ($(find "$T_DIR" -type f 2>/dev/null | wc -l) 个文件)"
fi

# --- 5) 只做 parquet 转换, 跳过它自己的下载/解包 ---
echo
echo ">>> 转换 train.jsonl -> train.parquet (--skip-download)"
cd "$VOPD_ROOT"
"$VOPD_PY" scripts/prepare_data.py --data-dir "$DATA_DIR" --skip-download

# --- 6) 检查产物 ---
echo
echo ">>> 产物检查"
echo -n "  train.parquet : "
[ -f "$DATA_DIR/train.parquet" ] && du -h "$DATA_DIR/train.parquet" | cut -f1 || { echo "❌ 未生成"; exit 1; }
echo -n "  数据目录总大小: "; du -sh "$DATA_DIR" 2>/dev/null | cut -f1

"$VOPD_PY" - "$DATA_DIR/train.parquet" <<'PY'
import sys, pathlib
import pyarrow.parquet as pq
t = pq.read_table(sys.argv[1])
print(f"\n  样本数: {t.num_rows}")
print(f"  列    : {t.column_names}")
row = t.slice(0, 1).to_pylist()[0]
print("\n  首条样本:")
for k, v in row.items():
    s = repr(v)
    print(f"    {k:24} {s[:120]}{'...' if len(s) > 120 else ''}")
# 训练脚本用 image_key=images / teacher_image_key=bbox_images, 确认对得上
for key in ("images", "bbox_images"):
    print(f"  [检查] 列 '{key}' {'存在 ✅' if key in t.column_names else '缺失 ❌ (训练脚本需要)'}")
PY
echo
echo ">>> 完成。下一步: 停掉 vllm server, 跑训练冒烟测试"
