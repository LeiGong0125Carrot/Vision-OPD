"""复现 TreeVGR-7B 在 TreeBench 上的官方推理结果。

基于 TreeVGR/inference_treebench.py 改写, 保持以下部分与官方逐字一致:
  - system prompt / user prompt 拼接 (OCR 类不拼选项)
  - 预填 "<think>" 强制进入思考
  - 生成参数 (top_p=0.001, top_k=1, temperature=0.01, max_new_tokens=1024)
  - <box>/<answer> 解析与 IoU 计算 (每个 GT 框取最优预测框, 对 GT 求平均)
  - min_pixels/max_pixels
改动:
  - 从本地 TSV 读数据 (data/TreeBench/TreeBench.tsv), 不走 HF hub
  - 串行单卡循环, 去掉 multiprocessing (官方是每 GPU 一个进程)
  - 逐条落盘 JSONL (含原始输出文本), 支持断点续跑
  - ATTN_IMPL 环境变量: rtxpro6000 用 flash_attention_2, B200 上 pip 的
    flash-attn 没有 sm_100 cubin, 需退到 sdpa

官方 README 预期: Overall 200/405=49.38, Mean IoU 43.3
"""
import ast
import json
import os
import re

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

VOPD_ROOT = os.environ.get("VOPD_ROOT", "/scratch/nkw3mr/Vision-OPD")
TSV_PATH = os.environ.get("TSV_PATH", f"{VOPD_ROOT}/data/TreeBench/TreeBench.tsv")
MODEL_PATH = os.environ.get("MODEL_PATH", "HaochenWang/TreeVGR-7B")
ATTN_IMPL = os.environ.get("ATTN_IMPL", "flash_attention_2")
OUT_PATH = os.environ.get(
    "OUT_PATH", f"{VOPD_ROOT}/eval/model_answer/treebench/treevgr-7b-repro_answer.jsonl"
)

SYSTEM_PROMPT = """A conversation between user and assistant. The user asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When referring to particular objects in the reasoning process, the assistant MUST localize the object with bounding box coordinates between <box> and </box>. You MUST strictly follow the format."""


# ---- 与官方 compute_box_iou 完全一致 ----
def compute_box_iou(predict_str: str, target_boxes: list) -> float:
    pattern = r"<box>(.*?)</box>"
    matches = re.findall(pattern, predict_str, re.DOTALL)

    all_boxes = []
    for match in matches:
        box = match.strip()
        coord_pattern = r"\[(\d+),(\d+),(\d+),(\d+)\]"
        coord_match = re.match(coord_pattern, box)
        if coord_match:
            x1, y1, x2, y2 = map(int, coord_match.groups())
            if x1 < x2 and y1 < y2:
                all_boxes.append([x1, y1, x2, y2])

    def compute_iou(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    if len(target_boxes) == 0:
        return 0.0
    total_iou = 0.0
    for t in target_boxes:
        total_iou += max((compute_iou(t, p) for p in all_boxes), default=0.0)
    return total_iou / len(target_boxes)


def build_messages(item):
    if item["category"] == "OCR":
        qs = item["question"]
    else:
        qs = item["question"] + " Options:\n" + item["multi-choice options"]
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{item['image']}"},
                {
                    "type": "text",
                    "text": qs
                    + "\nSelect the best answer to the above multiple-choice question based on the image. After the reasoning process, respond with only the letter of the correct option between <answer> and </answer>.",
                },
            ],
        },
    ]


def summarize(data):
    tags = [
        "Perception/Attributes", "Perception/Material", "Perception/Physical State",
        "Perception/Object Retrieval", "Perception/OCR",
        "Reasoning/Perspective Transform", "Reasoning/Ordering",
        "Reasoning/Contact and Occlusion", "Reasoning/Spatial Containment",
        "Reasoning/Comparison",
    ]
    total = correct = 0
    for tag in tags:
        c = sum(1 for x in data if x["category"] == tag and x["prediction"].upper() == x["answer"].upper())
        t = sum(1 for x in data if x["category"] == tag)
        total += t
        correct += c
        print(tag, f"{c}/{t}={round(c / t * 100, 2) if t else 0}")
    print("==> Overall", f"{correct}/{total}={round(correct / total * 100, 2)}")
    iou = np.array([x["iou"] for x in data])
    print("==> Mean IoU:", round(np.mean(iou) * 100, 2))


if __name__ == "__main__":
    df = load_dataset("csv", data_files=TSV_PATH, delimiter="\t")["train"]
    print(f"loaded {len(df)} rows from {TSV_PATH}")

    # 断点续跑: 已有结果按 index 跳过
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    done = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            for line in f:
                r = json.loads(line)
                done[r["index"]] = r
        print(f"resume: {len(done)} rows already done in {OUT_PATH}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPL,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, min_pixels=1280 * 28 * 28, max_pixels=16384 * 28 * 28
    )

    out_f = open(OUT_PATH, "a")
    for item in tqdm(df, desc="TreeBench"):
        if item["index"] in done:
            continue
        messages = build_messages(item)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text += "<think>"
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                top_p=0.001,
                top_k=1,
                temperature=0.01,
                repetition_penalty=1.0,
                max_new_tokens=1024,
                use_cache=True,
                do_sample=True,
            )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        box_iou = compute_box_iou(output_text, ast.literal_eval(item["target_instances"]))
        m = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
        ans = m.group(1).strip().upper() if m else output_text

        rec = {
            "index": item["index"],
            "category": item["category"],
            "answer": item["answer"],
            "prediction": ans,
            "iou": box_iou,
            "output": output_text,
        }
        done[item["index"]] = rec
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()

    summarize(list(done.values()))
