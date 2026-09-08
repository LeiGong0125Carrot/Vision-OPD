#!/usr/bin/env python
"""为 OPD-Aha 复刻生成 null 参照图 + 追加 null_images 列.

对 train_sa4k_hide.parquet 每行的 bbox_images[0] (hide 特权图) 生成同尺寸
mean-RGB 纯色空白图 (论文 revision_opd 的 I⁰ 同款, 复用
eval/treebench_probe/probe_hidden_response.py 的 null_image 逻辑):
  I⁰ = Image.new("RGB", size, mean_rgb(I⁺))

输出:
  data/TreeVGR-RL-37K/teacher_images_null/<id>.jpg   (4000 张)
  data/TreeVGR-RL-37K/train_sa4k_aha.parquet          (原 8 列 + null_images)

同尺寸保证 null 侧视觉 token 数与特权侧一致 → u = log p⁺ − log p⁰ 的
per-token 对齐不受视觉长度差干扰.

用法: $VOPD_PY opsa/aha/prep_null_images.py [--workers 16]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO, "data", "TreeVGR-RL-37K")
SRC_PARQUET = os.path.join(DATA_DIR, "train_sa4k_hide.parquet")
OUT_PARQUET = os.path.join(DATA_DIR, "train_sa4k_aha.parquet")
NULL_DIR = os.path.join(DATA_DIR, "teacher_images_null")


def make_null(src_path: str) -> str:
    dst = os.path.join(NULL_DIR, os.path.basename(src_path))
    if os.path.exists(dst):
        return dst
    with Image.open(src_path) as im:
        rgb = im.convert("RGB")
        arr = np.asarray(rgb)
        mean = tuple(int(v) for v in arr.reshape(-1, 3).mean(0))
        Image.new("RGB", rgb.size, mean).save(dst, quality=95)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(NULL_DIR, exist_ok=True)
    df = pd.read_parquet(SRC_PARQUET)
    srcs = [row[0]["path"] for row in df["bbox_images"]]
    assert all(len(row) == 1 for row in df["bbox_images"]), "每行应恰好 1 张特权图"

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        dsts = list(ex.map(make_null, srcs, chunksize=64))

    # 抽查: 尺寸一致 + 纯色
    for i in np.random.RandomState(0).choice(len(srcs), 20, replace=False):
        with Image.open(srcs[i]) as a, Image.open(dsts[i]) as b:
            assert a.size == b.size, f"尺寸不一致: {srcs[i]}"
            bar = np.asarray(b.convert("RGB"))
            assert bar.std(axis=(0, 1)).max() < 3.0, f"非纯色(jpeg噪声超限): {dsts[i]}"

    df["null_images"] = [[{"path": p}] for p in dsts]
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"OK: {len(df)} rows -> {OUT_PARQUET}")
    print(f"null images: {NULL_DIR} ({len(set(dsts))} unique)")


if __name__ == "__main__":
    main()
