"""Arm A 训练集: Vision-OPD-6K 过滤池 (高清>=1000vtok, 框占比>=1%, 剔除计数).

学生视角 = 干净原图 (original_images, 无红框无 focus 句) + 干净问题
教师视角 = hide 冠军渲染 (外扩2.42/纯黑底/画红框) + 同一问题
schema 与 train_sa4k_hide.parquet 完全一致; 标签仅存 reward_model/extra_info (训练不消费).

输出:
  data/TreeVGR-RL-37K/train_6k_armA_hide.parquet
  data/teacher_images_hide6k/{row}.jpg
"""
import json
import os
import re
import sys

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval", "treebench_probe"))
from infer_privilege import hide_compose

Image.MAX_IMAGE_PIXELS = None
ROOT_ABS = "/sfs/weka/scratch/nkw3mr/Vision-OPD"
ROOT = os.environ.get("VOPD_ROOT", "/scratch/nkw3mr/Vision-OPD")
OUT_IMG = f"{ROOT}/data/teacher_images_hide6k"
OUT_PARQUET = f"{ROOT}/data/TreeVGR-RL-37K/train_6k_armA_hide.parquet"
os.makedirs(OUT_IMG, exist_ok=True)
# 注: Qwen3.5 patch16+2x2merge, 图像 token ≈ W*H/1024; 全集最大图 ≈6.1k token,
# 训练侧用 MAX_PROMPT_LENGTH=8192 覆盖尾部, 不降采样 (保持与 TreeBench 体制一致).


def fine(q):
    ql = q.lower()
    if re.search(r"how many|count", ql):
        return "counting"
    if re.search(r"\b(left|right)\b", ql):
        return "spatial_lr"
    if re.search(r"\b(behind|front|above|below|top|bottom|under|over)\b", ql):
        return "spatial_fb"
    if re.search(r"\b(color|colour)\b", ql):
        return "color"
    if re.search(r"\b(material|made of)\b", ql):
        return "material"
    if re.search(r"\b(shape|geometric)\b", ql):
        return "shape"
    if re.search(r"\b(text|say|written|word|letter|number on|read)\b", ql):
        return "ocr"
    return "attribute_other"


rows = [json.loads(l) for l in open(f"{ROOT}/data/train.jsonl")]
recs, skip = [], {"lowres": 0, "smallbox": 0, "counting": 0}
for i, r in enumerate(rows):
    b = r["bbox"]
    src = f"{ROOT}/data/{r['original_images'][0]}"
    img = Image.open(src)
    W, H = img.size
    if W * H // (28 * 28 * 4) < 1000:
        skip["lowres"] += 1
        continue
    frac = max(0, b[2] - b[0]) * max(0, b[3] - b[1]) / (W * H) * 100
    if frac < 1.0:
        skip["smallbox"] += 1
        continue
    q = r["extra_info"]["question"]
    task = fine(q)
    if task == "counting":
        skip["counting"] += 1
        continue
    dst = f"{OUT_IMG}/{i}.jpg"
    if not os.path.exists(dst):
        hide_compose(img.convert("RGB"), [b]).save(dst, quality=92)
    msg = [{"content": f"<image>\n{q}", "role": "user"}]
    recs.append({
        "data_source": "visionopd6k_armA",
        "prompt": msg,
        "images": [{"path": f"{ROOT_ABS}/data/{r['original_images'][0]}"}],
        "bbox_images": [{"path": f"{ROOT_ABS}/data/teacher_images_hide6k/{i}.jpg"}],
        "teacher_prompt": msg,
        "ability": "visual_question_answering",
        "reward_model": {"ground_truth": r["extra_info"]["answer"], "style": "none"},
        "extra_info": {"answer": r["extra_info"]["answer"], "n_boxes": 1, "orig_row": i,
                       "question": q, "task": task, "box_frac": round(frac, 3)},
    })
    if len(recs) % 500 == 0:
        print(len(recs), "...", flush=True)

df = pd.DataFrame(recs)
df.to_parquet(OUT_PARQUET)
import collections
print(f"完成: {len(df)} 行 -> {OUT_PARQUET}")
print("过滤:", skip)
print("题型:", dict(collections.Counter(r["extra_info"]["task"] for r in recs)))
