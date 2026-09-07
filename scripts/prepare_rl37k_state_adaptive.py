"""subset4k_adv_v2 -> State-Adaptive OPD 训练数据。

teacher 特权形态 = region (2.42x 外扩裁剪 + 红框, 每框一张 crop 替换整图) ——
RL-37K 过户验证的冠军形态 (逐任务类 ≈2x hide, win 91%)。
文本两侧全裸: student prompt = 原 problem (<image>+纯问题), teacher_prompt =
n 个 <image> 占位 + 同一纯问题, 无任何特权句 / 标签句。

输出:
  data/TreeVGR-RL-37K/teacher_images_region/{orig_row}_{k}.jpg
  data/TreeVGR-RL-37K/train_sa4k.parquet   (schema 对齐 prepare_data.py)

运行:
  /scratch/nkw3mr/envs/vision-opd/bin/python scripts/prepare_rl37k_state_adaptive.py
"""
import os
import sys

import datasets
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval", "treebench_probe"))
from infer_privilege import REGION_EXPAND, draw_gt  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data", "TreeVGR-RL-37K")
SRC = os.path.join(DATA_DIR, "subset4k_adv_v2.parquet")
TEACHER_DIR = os.path.join(DATA_DIR, "teacher_images_region")
OUT = os.path.join(DATA_DIR, "train_sa4k.parquet")
MIN_CROP = 28  # Qwen patch 下限保险: 过小裁剪对称扩到 28px


def region_crop_bounded(pil_img, box):
    """region_crop (画红框 -> 2.42x 外扩裁剪) + 最小尺寸保险。"""
    w, h = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * (REGION_EXPAND - 1) / 2, bh * (REGION_EXPAND - 1) / 2
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    if cx2 - cx1 < MIN_CROP:
        pad = (MIN_CROP - (cx2 - cx1) + 1) // 2
        cx1, cx2 = max(0, cx1 - pad), min(w, cx2 + pad)
    if cy2 - cy1 < MIN_CROP:
        pad = (MIN_CROP - (cy2 - cy1) + 1) // 2
        cy1, cy2 = max(0, cy1 - pad), min(h, cy2 + pad)
    anno = draw_gt(pil_img, [box])
    return anno.crop((cx1, cy1, cx2, cy2))


def main():
    os.makedirs(TEACHER_DIR, exist_ok=True)
    df = pd.read_parquet(SRC)
    print(f"source rows: {len(df)}")

    records, n_clip, n_skip = [], 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="render"):
        orig = int(row["orig_row"])
        img_path = os.path.join(DATA_DIR, row["images"][0])
        if not os.path.exists(img_path):
            n_skip += 1
            continue
        pil = Image.open(img_path).convert("RGB")
        w, h = pil.size

        boxes = []
        for inst in row["target_instances"]:
            b = [float(v) for v in inst["bbox"]]
            cb = [max(0.0, min(w - 1, b[0])), max(0.0, min(h - 1, b[1])),
                  max(0.0, min(w, b[2])), max(0.0, min(h, b[3]))]
            if cb != b:
                n_clip += 1
            if cb[2] - cb[0] < 2 or cb[3] - cb[1] < 2:
                continue
            boxes.append(cb)
        if not boxes:
            n_skip += 1
            continue

        teacher_paths = []
        for k, b in enumerate(boxes):
            tp = os.path.join(TEACHER_DIR, f"{orig}_{k}.jpg")
            if not os.path.exists(tp):
                region_crop_bounded(pil, b).save(tp, quality=92)
            teacher_paths.append(tp)

        problem = str(row["problem"])
        question = problem.replace("<image>", "").strip()
        answer = str(row["answer"]).strip()

        records.append({
            "data_source": "treevgr_rl37k_state_adaptive",
            "prompt": [{"role": "user", "content": problem}],
            "images": [{"path": img_path}],
            "bbox_images": [{"path": p} for p in teacher_paths],
            "teacher_prompt": [{"role": "user",
                                "content": "".join(["<image>"] * len(teacher_paths)) + "\n" + question}],
            "ability": "visual_question_answering",
            "reward_model": {"style": "none", "ground_truth": answer},
            "extra_info": {
                "answer": answer,
                "question": question,
                "task": str(row["task"]),
                "orig_row": orig,
                "n_boxes": len(boxes),
            },
        })

    print(f"records: {len(records)}  (skipped {n_skip}, clipped-box rows {n_clip})")
    nb = pd.Series([r["extra_info"]["n_boxes"] for r in records])
    print("n_boxes distribution:\n", nb.value_counts().sort_index())
    ds = datasets.Dataset.from_list(records)
    ds.to_parquet(OUT)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
