"""用 VLM(默认 32B)给 TreeBench 的 GT bbox 打标签(伪标签, staged 精神: 模型自产)。

TreeBench 的 target_instances 只有坐标没标签, 特权实验需要"框↔对象"绑定。
每题一次调用: 原图 + GT 框(转 [0,1000] 归一化, Qwen3-VL 母语) -> 每框一个短语标签。
输出 eval/treebench_gt_labels.jsonl: {index, gt_boxes(绝对像素), gt_boxes_norm, labels, raw}
标签质量可人工抽查: treebench_cases/<idx>/crop_gt_<i>.jpg 与 labels[i] 对照。

运行 (vision-opd 环境, 需 GPU):
  python label_gt_boxes.py --model Qwen/Qwen3-VL-32B-Instruct --limit 10   # 冒烟
"""
import argparse
import ast
import base64
import io
import json
import os
import re

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

VOPD_ROOT = os.environ.get("VOPD_ROOT", "/scratch/nkw3mr/Vision-OPD")
TSV_PATH = os.environ.get("TSV_PATH", f"{VOPD_ROOT}/data/TreeBench/TreeBench.tsv")


def build_prompt(item, nb):
    regions = "\n".join(f'{i}: {{"bbox_2d": [{b[0]}, {b[1]}, {b[2]}, {b[3]}]}}' for i, b in enumerate(nb))
    # 给题目文本做上下文, 让标签用题目的指称词汇 (绑定实验需要的正是这种一致性)
    return (f"Question about this image (for context only, do NOT answer it):\n{item['question']}\n\n"
            f"The following regions are marked in the image, with bbox_2d in [x1,y1,x2,y2] "
            f"normalized to 0-1000:\n{regions}\n\n"
            f"For EACH region, identify the object it contains, in a short phrase, preferring the "
            f"wording used in the question when it matches. Output ONLY a JSON array of strings, "
            f"one label per region, in the same order.")


def parse_labels(text, n):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return (arr + [""] * n)[:n]
        except json.JSONDecodeError:
            pass
    # 兜底: 按行抓引号串
    qs = re.findall(r'"([^"]{1,80})"', text)
    return (qs + [""] * n)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--out", default=f"{VOPD_ROOT}/eval/treebench_gt_labels.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = load_dataset("csv", data_files=TSV_PATH, delimiter="\t")["train"]
    if args.limit:
        df = df.select(range(args.limit))

    done = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                done[r["index"]] = r
        print(f"resume: {len(done)} rows already done")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)

    out_f = open(args.out, "a")
    for item in tqdm(df, desc="label-gt"):
        if item["index"] in done:
            continue
        w, h = Image.open(io.BytesIO(base64.b64decode(item["image"]))).size
        gt = ast.literal_eval(item["target_instances"])
        nb = [[round(b[0] * 1000 / w), round(b[1] * 1000 / h),
               round(b[2] * 1000 / w), round(b[3] * 1000 / h)] for b in gt]

        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{item['image']}"},
            {"type": "text", "text": build_prompt(item, nb)},
        ]}]
        image_inputs, _ = process_vision_info(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=image_inputs, padding=True,
                           return_tensors="pt").to(model.device)
        with torch.inference_mode():
            gen = model.generate(**inputs, top_p=0.001, top_k=1, temperature=0.01,
                                 repetition_penalty=1.0, max_new_tokens=256,
                                 use_cache=True, do_sample=True)
        raw = processor.batch_decode([gen[0][inputs.input_ids.shape[1]:]],
                                     skip_special_tokens=True)[0]
        labels = parse_labels(raw, len(gt))

        rec = {"index": item["index"], "category": item["category"],
               "gt_boxes": gt, "gt_boxes_norm": nb, "labels": labels,
               "labeler": args.model, "raw": raw}
        done[item["index"]] = rec
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()

    empty = sum(1 for r in done.values() for x in r["labels"] if not x)
    total = sum(len(r["labels"]) for r in done.values())
    print(f"done: {len(done)} 题, {total} 框, 空标签 {empty}")


if __name__ == "__main__":
    main()
