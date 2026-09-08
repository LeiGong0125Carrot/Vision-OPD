#!/usr/bin/env python
"""臂 B: 共享上下文特权对 (p⁺=全图+GT crops vs p⁰=全图+同尺寸空白块).

对齐论文的对比结构: 两侧 teacher 都看到学生的全图, 只在"证据 crop vs 空白块"上
不同 → u 按构造只装证据贡献, hide 形态的视野错配通道 (u_mean≈−0.4) 归零。

数据源:
  train_sa4k_aha.parquet         (4000 行; extra_info.orig_row 指向源表行号)
  vstar30k_visdrone6k_x1y1x2y2.parquet (target_instances: 绝对像素 x1y1x2y2)

输出:
  teacher_images_crop/{orig_row}_{i}.jpg      GT 区域裁剪 (短边 <56 时等比放大到 56)
  teacher_images_cropnull/{orig_row}_{i}.jpg  同尺寸 mean-RGB 空白块
  train_sa4k_pair.parquet:
    bbox_images  -> [全图, crop_0, ...]      (沿用列名, launcher 只需换 TASK_TRAIN_FILE)
    null_images  -> [全图, null_0, ...]
    teacher_prompt -> 1+n_crops 个 <image> 占位符

用法: $VOPD_PY opsa/aha/prep_pair_images.py [--workers 16]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO, "data", "TreeVGR-RL-37K")
AHA_PARQUET = os.path.join(DATA_DIR, "train_sa4k_aha.parquet")
SRC_PARQUET = os.path.join(DATA_DIR, "vstar30k_visdrone6k_x1y1x2y2.parquet")
OUT_PARQUET = os.path.join(DATA_DIR, "train_sa4k_pair.parquet")
CROP_DIR = os.path.join(DATA_DIR, "teacher_images_crop")
NULL_DIR = os.path.join(DATA_DIR, "teacher_images_cropnull")
MIN_SIDE = 56          # Qwen 视觉 patch 28 的 2 倍, 防止微小框被压没
PAD_FRAC = 0.1         # 裁剪时四周各扩 10% 上下文余量


def make_crops(job):
    orig_row, img_path, bboxes = job
    outs = []
    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
        W, H = rgb.size
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            pw, ph = (x2 - x1) * PAD_FRAC, (y2 - y1) * PAD_FRAC
            a, b = max(0, int(x1 - pw)), max(0, int(y1 - ph))
            c, d = min(W, int(x2 + pw)), min(H, int(y2 + ph))
            if c - a < 2 or d - b < 2:   # 退化框: 以中心取 8x8 最小窗
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                a, b = max(0, cx - 4), max(0, cy - 4)
                c, d = min(W, cx + 4), min(H, cy + 4)
            crop = rgb.crop((a, b, c, d))
            if min(crop.size) < MIN_SIDE:
                s = MIN_SIDE / min(crop.size)
                crop = crop.resize((max(MIN_SIDE, int(crop.width * s)),
                                    max(MIN_SIDE, int(crop.height * s))), Image.BICUBIC)
            cp = os.path.join(CROP_DIR, f"{orig_row}_{i}.jpg")
            npth = os.path.join(NULL_DIR, f"{orig_row}_{i}.jpg")
            if not os.path.exists(cp):
                crop.save(cp, quality=95)
            if not os.path.exists(npth):
                mean = tuple(int(v) for v in np.asarray(crop).reshape(-1, 3).mean(0))
                Image.new("RGB", crop.size, mean).save(npth, quality=95)
            outs.append((cp, npth))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(CROP_DIR, exist_ok=True)
    os.makedirs(NULL_DIR, exist_ok=True)

    df = pd.read_parquet(AHA_PARQUET)
    src = pd.read_parquet(SRC_PARQUET)

    jobs, metas = [], []
    for ridx, row in df.iterrows():
        orig_row = int(row["extra_info"]["orig_row"])
        img_path = row["images"][0]["path"]
        insts = src.iloc[orig_row]["target_instances"]
        bboxes = [tuple(float(v) for v in inst["bbox"]) for inst in insts]
        assert bboxes, f"row {ridx}: 无 target_instances"
        jobs.append((orig_row, img_path, bboxes))
        metas.append((ridx, orig_row, img_path, row["extra_info"]["question"]))

    if args.workers <= 1:
        all_outs = [make_crops(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            all_outs = list(ex.map(make_crops, jobs, chunksize=32))

    pair_imgs, pair_nulls, prompts = [], [], []
    n_crop_hist = {}
    for (ridx, orig_row, img_path, question), outs in zip(metas, all_outs):
        crops = [{"path": p} for p, _ in outs]
        nulls = [{"path": p} for _, p in outs]
        pair_imgs.append([{"path": img_path}] + crops)
        pair_nulls.append([{"path": img_path}] + nulls)
        ph = " ".join(["<image>"] * len(crops))
        prompts.append([{
            "role": "user",
            "content": f"<image>\nZoomed-in views of the key regions: {ph}\n{question}",
        }])
        n_crop_hist[len(crops)] = n_crop_hist.get(len(crops), 0) + 1

    df["bbox_images"] = pair_imgs
    df["null_images"] = pair_nulls
    df["teacher_prompt"] = prompts
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"OK: {len(df)} rows -> {OUT_PARQUET}")
    print(f"crops per row: {dict(sorted(n_crop_hist.items()))}")

    # 抽查: 占位符数 == 图片数; crop 文件存在且尺寸达标
    chk = pd.read_parquet(OUT_PARQUET)
    for i in np.random.RandomState(0).choice(len(chk), 25, replace=False):
        r = chk.iloc[i]
        n_ph = r["teacher_prompt"][0]["content"].count("<image>")
        assert n_ph == len(r["bbox_images"]) == len(r["null_images"]), f"row {i}: 占位符/图片数不匹配"
        for item in list(r["bbox_images"][1:]) + list(r["null_images"][1:]):
            with Image.open(item["path"]) as im:
                assert min(im.size) >= MIN_SIDE, f"{item['path']} 尺寸不达标"
    print("抽查通过")


if __name__ == "__main__":
    main()
