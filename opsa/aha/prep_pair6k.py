#!/usr/bin/env python
"""高清臂 B: 共享上下文特权对, 6karmA 版 (2459 题, 标准训练集).

与 4k 版 (prep_pair_images.py) 的区别: bbox 不走 orig_row 反查 (跨表坐标系有风险),
而是**直接从 hide6k 教师图提取非黑区域**:
  hide 图 = 原图尺寸黑底 + GT 框内容原位合成 (构建时已加 ~2.45x 线性上下文余量,
  贴边裁剪)。非黑 bbox 即 crop 区域, 与 hide 教师所见信息完全一致, 坐标系零风险。
  阈值稳定性已验证 (th=10/30/60 bbox 基本不变)。

输出:
  teacher_images_crop6k/{i}.jpg      非黑区域裁剪 (自原图, 短边保底 56)
  teacher_images_cropnull6k/{i}.jpg  同尺寸 mean-RGB 空白块
  train_6karmA_pair.parquet:
    bbox_images -> [原图, crop], null_images -> [原图, null 块],
    teacher_prompt -> 2 个 <image> 占位符 (全部 n_boxes=1, 已验证)

用法: $VOPD_PY opsa/aha/prep_pair6k.py
"""

import os

import numpy as np
import pandas as pd
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO, "data", "TreeVGR-RL-37K")
SRC_PARQUET = os.path.join(DATA_DIR, "train_6k_armA_hide.parquet")
OUT_PARQUET = os.path.join(DATA_DIR, "train_6karmA_pair.parquet")
CROP_DIR = os.path.join(DATA_DIR, "teacher_images_crop6k")
NULL_DIR = os.path.join(DATA_DIR, "teacher_images_cropnull6k")
MIN_SIDE = 56
NZ_THRESHOLD = 30   # RGB 和阈值; 已验证对 bbox 范围不敏感


def extract_box(hide_path):
    with Image.open(hide_path) as im:
        arr = np.asarray(im.convert("RGB")).astype(int)
    nz = arr.sum(axis=2) > NZ_THRESHOLD
    if not nz.any():
        return None
    ys, xs = np.where(nz)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def main():
    os.makedirs(CROP_DIR, exist_ok=True)
    os.makedirs(NULL_DIR, exist_ok=True)
    df = pd.read_parquet(SRC_PARQUET)

    pair_imgs, pair_nulls, prompts = [], [], []
    for i, row in df.iterrows():
        orig_path = row["images"][0]["path"]
        box = extract_box(row["bbox_images"][0]["path"])
        assert box is not None, f"row {i}: hide 图全黑"
        cp = os.path.join(CROP_DIR, f"{i}.jpg")
        npth = os.path.join(NULL_DIR, f"{i}.jpg")
        if not (os.path.exists(cp) and os.path.exists(npth)):
            with Image.open(orig_path) as im:
                crop = im.convert("RGB").crop(box)
            if min(crop.size) < MIN_SIDE:
                s = MIN_SIDE / min(crop.size)
                crop = crop.resize((max(MIN_SIDE, int(crop.width * s)),
                                    max(MIN_SIDE, int(crop.height * s))), Image.BICUBIC)
            crop.save(cp, quality=95)
            mean = tuple(int(v) for v in np.asarray(crop).reshape(-1, 3).mean(0))
            Image.new("RGB", crop.size, mean).save(npth, quality=95)

        question = row["extra_info"]["question"]
        pair_imgs.append([{"path": orig_path}, {"path": cp}])
        pair_nulls.append([{"path": orig_path}, {"path": npth}])
        prompts.append([{
            "role": "user",
            "content": f"<image>\nZoomed-in view of the key region: <image>\n{question}",
        }])
        if (i + 1) % 500 == 0:
            print(f"{i+1}/{len(df)}")

    df["bbox_images"] = pair_imgs
    df["null_images"] = pair_nulls
    df["teacher_prompt"] = prompts
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"OK: {len(df)} rows -> {OUT_PARQUET}")

    chk = pd.read_parquet(OUT_PARQUET)
    for i in np.random.RandomState(0).choice(len(chk), 20, replace=False):
        r = chk.iloc[i]
        assert r["teacher_prompt"][0]["content"].count("<image>") == 2 == len(r["bbox_images"]) == len(r["null_images"])
        with Image.open(r["bbox_images"][1]["path"]) as a, Image.open(r["null_images"][1]["path"]) as b:
            assert a.size == b.size and min(a.size) >= MIN_SIDE
    print("抽查通过")


if __name__ == "__main__":
    main()
