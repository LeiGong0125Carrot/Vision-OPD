"""Qwen3-VL 系列在 TreeBench 上的 grounded probe: 先生成 bbox 再推理。

设计原则 (2026-08-28 与 TreeVGR 复现对齐后确定):
  - system prompt 与 TreeVGR 官方推理**逐字一致** —— 它不规定坐标空间,
    每个模型用自己的原生坐标习惯 (Qwen2.5-VL 绝对像素 / Qwen3-VL [0,1000] 归一化),
    坐标空间在打分端处理, 不在 prompt 里处理。
  - user turn 拼接规则 / OCR 不拼选项 / assistant 端预填 "<think>" / 贪心生成参数 /
    <answer> 抽取正则, 全部与 TreeVGR 官方 inference_treebench.py 一致。
  - min/max pixels 用各模型 processor 的**默认值** (不套 TreeVGR 的设置), 实际值记录进结果。
  - 打分: 每题同时计算 iou_abs (框按绝对像素) 和 iou_norm (框按 [0,1000] 归一化后
    乘 w/1000, h/1000 还原), 汇总时按整个 run 的坐标越界率自动判定该模型的坐标空间
    (任一坐标 >1000 的框占比 >2% 判为绝对像素), 也可用 --coord-space 强制指定。

运行 (vision-opd 环境, transformers 5.5; Qwen3-VL 需要新 transformers, treevgr 环境的 4.52 不支持):
  python infer_grounded.py --model Qwen/Qwen3-VL-4B-Instruct
  python infer_grounded.py --model Qwen/Qwen3-VL-4B-Instruct --limit 10   # 冒烟
支持断点续跑: 输出 jsonl 已有的 index 会跳过。

注意 transformers 5.x 的 apply_chat_template 会原地改写 messages 里的 image_url 字段,
因此必须先 process_vision_info 再 apply_chat_template (顺序与官方脚本相反, 结果等价)。
"""
import argparse
import ast
import base64
import io
import json
import os
import re

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

VOPD_ROOT = os.environ.get("VOPD_ROOT", "/scratch/nkw3mr/Vision-OPD")
TSV_PATH = os.environ.get("TSV_PATH", f"{VOPD_ROOT}/data/TreeBench/TreeBench.tsv")

# —— 与 TreeVGR 官方逐字一致 ——
SYSTEM_PROMPT = """A conversation between user and assistant. The user asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When referring to particular objects in the reasoning process, the assistant MUST localize the object with bounding box coordinates between <box> and </box>. You MUST strictly follow the format."""

USER_SUFFIX = "\nSelect the best answer to the above multiple-choice question based on the image. After the reasoning process, respond with only the letter of the correct option between <answer> and </answer>."

# base 模型不认识 <box> 语法 (TreeVGR 是 SFT+RL 教出来的; 论文里 base Qwen2.5-VL 的
# mIoU 也是 "–")。冒烟发现 4B 对 system 指令服从度低 (verbatim/example 都零框),
# 但 user turn 的 <answer> 指令被执行 —— userfmt 把格式要求复述进 user turn。
# 变体 = (system prompt, 追加到 user 文本末尾的格式要求)
PROMPT_VARIANTS = {
    "verbatim": (SYSTEM_PROMPT, ""),
    "example": (SYSTEM_PROMPT + " Bounding boxes must be written in the format <box>[x1,y1,x2,y2]</box>.", ""),
    "userfmt": (SYSTEM_PROMPT,
                " In your reasoning, you MUST provide the bounding box of every object you refer to, in the format <box>[x1,y1,x2,y2]</box>."),
}

TAGS = ["Perception/Attributes", "Perception/Material", "Perception/Physical State",
        "Perception/Object Retrieval", "Perception/OCR",
        "Reasoning/Perspective Transform", "Reasoning/Ordering",
        "Reasoning/Contact and Occlusion", "Reasoning/Spatial Containment",
        "Reasoning/Comparison"]


def parse_boxes(predict_str: str):
    """官方同款解析: <box>[x1,y1,x2,y2]</box>, 要求 x1<x2 且 y1<y2。"""
    boxes = []
    for m in re.findall(r"<box>(.*?)</box>", predict_str, re.DOTALL):
        cm = re.match(r"\[(\d+),(\d+),(\d+),(\d+)\]", m.strip())
        if cm:
            x1, y1, x2, y2 = map(int, cm.groups())
            if x1 < x2 and y1 < y2:
                boxes.append([x1, y1, x2, y2])
    return boxes


def avg_best_iou(pred_boxes, target_boxes):
    """官方口径: 每个 GT 框取所有预测框中 IoU 最优, 对 GT 框平均。"""
    def iou(b1, b2):
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0
    if not target_boxes:
        return 0.0
    return sum(max((iou(t, p) for p in pred_boxes), default=0.0) for t in target_boxes) / len(target_boxes)


def build_messages(item, variant="verbatim"):
    sys_prompt, user_extra = PROMPT_VARIANTS[variant]
    if item["category"] == "OCR":
        qs = item["question"]
    else:
        qs = item["question"] + " Options:\n" + item["multi-choice options"]
    return [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{item['image']}"},
            {"type": "text", "text": qs + USER_SUFFIX + user_extra},
        ]},
    ]


def summarize(data, coord_space_flag):
    # 坐标空间判定: 任一坐标 >1000 的框占比
    n_boxes = sum(len(x["pred_boxes"]) for x in data)
    n_over = sum(1 for x in data for b in x["pred_boxes"] if max(b) > 1000)
    if coord_space_flag != "auto":
        space = coord_space_flag
    else:
        space = "abs" if n_boxes and n_over / n_boxes > 0.02 else "norm1000"
    iou_key = "iou_abs" if space == "abs" else "iou_norm"
    print(f"\n坐标空间判定: {space} (越界框 {n_over}/{n_boxes}, 判定方式: {coord_space_flag})")

    total = correct = 0
    for tag in TAGS:
        g = [x for x in data if x["category"] == tag]
        c = sum(1 for x in g if x["prediction"].upper() == x["answer"].upper())
        total += len(g)
        correct += c
        if g:
            print(tag, f"{c}/{len(g)}={round(c / len(g) * 100, 2)}")
    print("==> Overall", f"{correct}/{total}={round(correct / total * 100, 2)}")
    print("==> Mean IoU:", round(float(np.mean([x[iou_key] for x in data])) * 100, 2),
          f"(按 {space}; 另一口径 {'iou_norm' if space == 'abs' else 'iou_abs'}:",
          round(float(np.mean([x['iou_norm' if space == 'abs' else 'iou_abs'] for x in data])) * 100, 2), ")")
    n_fmt = sum(1 for x in data if not re.search(r"<answer>(.*?)</answer>", x["output"], re.DOTALL))
    print(f"==> 无 <answer> 标签: {n_fmt}/{len(data)}   零框: {sum(1 for x in data if not x['pred_boxes'])}/{len(data)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id 或本地路径, 如 Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out", default=None, help="输出 jsonl, 默认按模型名生成")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题 (冒烟用)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--coord-space", choices=["auto", "abs", "norm1000"], default="auto")
    ap.add_argument("--sys-variant", choices=list(PROMPT_VARIANTS), default="verbatim",
                    help="verbatim=TreeVGR 原文; example=system 加格式示例; userfmt=user turn 加格式要求")
    args = ap.parse_args()

    tag = args.model.rstrip("/").split("/")[-1].lower()
    suffix = "grounded" if args.sys_variant == "verbatim" else f"grounded-{args.sys_variant}"
    out_path = args.out or f"{VOPD_ROOT}/eval/model_answer/treebench/{tag}-{suffix}_answer.jsonl"

    df = load_dataset("csv", data_files=TSV_PATH, delimiter="\t")["train"]
    if args.limit:
        df = df.select(range(args.limit))
    print(f"loaded {len(df)} rows; out -> {out_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done[r["index"]] = r
        print(f"resume: {len(done)} rows already done")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.model)  # min/max pixels 用模型默认
    ip = getattr(processor, "image_processor", None)
    pixel_cfg = {k: getattr(ip, k, None) for k in ("min_pixels", "max_pixels", "size")} if ip else {}
    print("processor pixel 设置:", pixel_cfg)

    out_f = open(out_path, "a")
    for item in tqdm(df, desc=tag):
        if item["index"] in done:
            continue
        messages = build_messages(item, args.sys_variant)
        # 先取视觉输入, 再渲染模板 (transformers 5.x 会原地改写 messages)
        image_inputs, video_inputs = process_vision_info(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text += "<think>"
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                top_p=0.001, top_k=1, temperature=0.01, repetition_penalty=1.0,
                max_new_tokens=args.max_new_tokens, use_cache=True, do_sample=True,
            )
        output_text = processor.batch_decode(
            [gen[0][inputs.input_ids.shape[1]:]],
            skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )[0]

        w, h = Image.open(io.BytesIO(base64.b64decode(item["image"]))).size
        gt_boxes = ast.literal_eval(item["target_instances"])
        pred_boxes = parse_boxes(output_text)
        pred_norm = [[b[0] * w / 1000, b[1] * h / 1000, b[2] * w / 1000, b[3] * h / 1000] for b in pred_boxes]

        m = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
        ans = m.group(1).strip().upper() if m else output_text

        rec = {
            "index": item["index"], "category": item["category"], "answer": item["answer"],
            "prediction": ans,
            "iou_abs": avg_best_iou(pred_boxes, gt_boxes),
            "iou_norm": avg_best_iou(pred_norm, gt_boxes),
            "pred_boxes": pred_boxes, "image_size": [w, h],
            "model": args.model, "condition": "grounded", "sys_variant": args.sys_variant,
            "pixel_cfg": {k: str(v) for k, v in pixel_cfg.items()},
            "output": output_text,
        }
        done[item["index"]] = rec
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()

    summarize(list(done.values()), args.coord_space)


if __name__ == "__main__":
    main()
