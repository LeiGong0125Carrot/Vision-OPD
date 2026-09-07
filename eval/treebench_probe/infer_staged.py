"""Qwen3-VL 两段式 probe: 先生成 bbox(原生 grounding 语式), 再推理。

背景 (见 README): 单次调用里要求 base Qwen3-VL 在推理中交错输出 <box> 基本不服从
(verbatim/example 零框, userfmt 仅 1/10 题出框), 所以拆成两段:
  Stage1 定位: 无 system prompt, 用 Qwen3-VL 原生的 JSON grounding 格式
              ({"bbox_2d": [x1,y1,x2,y2], "label": ...}) 让它框出题目相关对象。
  Stage2 推理: system = TreeVGR verbatim 删去 <box> 一句 (即 direct 条件的契约),
              user turn 注入 Stage1 的框 ("Relevant objects ..."), 预填 <think>。

--condition direct 跳过 Stage1 且不注入框, 其余完全相同 —— 两条件只差
"框信息是否在上下文里"一个变量。

IoU 用 Stage1 的框算 (双口径 iou_abs / iou_norm + 越界率自动判定, 同 infer_grounded.py)。
运行 (vision-opd 环境):
  python infer_staged.py --model Qwen/Qwen3-VL-4B-Instruct --condition staged --limit 10
  python infer_staged.py --model Qwen/Qwen3-VL-4B-Instruct --condition direct --limit 10
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

# TreeVGR system prompt 删去 <box> 那一句 (最小删改, think/answer 契约保留)
SYSTEM_PROMPT_NOBOX = """A conversation between user and assistant. The user asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. You MUST strictly follow the format."""

USER_SUFFIX = "\nSelect the best answer to the above multiple-choice question based on the image. After the reasoning process, respond with only the letter of the correct option between <answer> and </answer>."

STAGE1_SUFFIX = """
Do not answer the question yet. First, locate all objects in the image that are relevant to answering this question. Output ONLY a JSON array, where each element is {"bbox_2d": [x1, y1, x2, y2], "label": "<short description>"}."""

# 单轮先框后推: 输出开头 = 原生 grounding JSON (模型的训练分布), 然后接着推理作答
BOXFIRST_SUFFIX = """
Answer in exactly this order: (1) First, locate all objects in the image that are relevant to answering this question, and output them as a JSON array where each element is {"bbox_2d": [x1, y1, x2, y2], "label": "<short description>"}. (2) Then, reason step by step based on these located objects. (3) Finally, respond with only the letter of the correct option between <answer> and </answer>."""

TAGS = ["Perception/Attributes", "Perception/Material", "Perception/Physical State",
        "Perception/Object Retrieval", "Perception/OCR",
        "Reasoning/Perspective Transform", "Reasoning/Ordering",
        "Reasoning/Contact and Occlusion", "Reasoning/Spatial Containment",
        "Reasoning/Comparison"]


def question_text(item):
    if item["category"] == "OCR":
        return item["question"]
    return item["question"] + " Options:\n" + item["multi-choice options"]


def parse_stage1_boxes(text: str):
    """解析原生 grounding 输出: JSON 数组优先, 失败则逐个抓 bbox_2d。"""
    objs = []
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            for o in arr:
                if isinstance(o, dict) and isinstance(o.get("bbox_2d"), list) and len(o["bbox_2d"]) == 4:
                    objs.append({"bbox_2d": [float(v) for v in o["bbox_2d"]], "label": str(o.get("label", ""))})
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if not objs:
        for bm in re.finditer(r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
                              r'(?:\s*,\s*"label"\s*:\s*"([^"]*)")?', text):
            objs.append({"bbox_2d": [float(bm.group(i)) for i in range(1, 5)], "label": bm.group(5) or ""})
    return [o for o in objs if o["bbox_2d"][0] < o["bbox_2d"][2] and o["bbox_2d"][1] < o["bbox_2d"][3]]


def avg_best_iou(pred_boxes, target_boxes):
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


def summarize(data, coord_space_flag, condition):
    total = correct = 0
    for tag in TAGS:
        g = [x for x in data if x["category"] == tag]
        c = sum(1 for x in g if x["prediction"].upper() == x["answer"].upper())
        total += len(g)
        correct += c
        if g:
            print(tag, f"{c}/{len(g)}={round(c / len(g) * 100, 2)}")
    print("==> Overall", f"{correct}/{total}={round(correct / total * 100, 2)}")
    n_fmt = sum(1 for x in data if not re.search(r"<answer>(.*?)</answer>", x["output"], re.DOTALL))
    print(f"==> 无 <answer> 标签: {n_fmt}/{len(data)}")
    if condition in ("staged", "boxfirst"):
        n_boxes = sum(len(x["pred_boxes"]) for x in data)
        n_over = sum(1 for x in data for b in x["pred_boxes"] if max(b) > 1000)
        space = coord_space_flag if coord_space_flag != "auto" else (
            "abs" if n_boxes and n_over / n_boxes > 0.02 else "norm1000")
        k = "iou_abs" if space == "abs" else "iou_norm"
        other = "iou_norm" if space == "abs" else "iou_abs"
        print(f"==> 坐标空间判定: {space} (越界框 {n_over}/{n_boxes})")
        print("==> Mean IoU:", round(float(np.mean([x[k] for x in data])) * 100, 2),
              f"(按 {space}; 另一口径 {other}:", round(float(np.mean([x[other] for x in data])) * 100, 2), ")")
        print(f"==> Stage1 零框: {sum(1 for x in data if not x['pred_boxes'])}/{len(data)}")


def run_generate(model, processor, messages, max_new_tokens, prefill="", no_think=False):
    image_inputs, video_inputs = process_vision_info(messages)
    # Qwen3.5 的模板默认开 thinking (assistant 自动开在思考块里, 1024 预算会被吃穿);
    # no_think 时显式关掉, 与 Qwen3-VL-Instruct 非思考协议可比。只在需要时传 kwarg,
    # 避免动到不认识该参数的模板。
    tmpl_kw = {"enable_thinking": False} if no_think else {}
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **tmpl_kw) + prefill
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        gen = model.generate(**inputs, top_p=0.001, top_k=1, temperature=0.01,
                             repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                             use_cache=True, do_sample=True)
    return processor.batch_decode([gen[0][inputs.input_ids.shape[1]:]],
                                  skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", choices=["staged", "direct", "boxfirst"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--stage1-max-new-tokens", type=int, default=512)
    ap.add_argument("--coord-space", choices=["auto", "abs", "norm1000"], default="auto")
    # 8B-Instruct 对预填的 <think> 直接输出 EOS (263/405 空回复), 预填是 TreeVGR 的习惯,
    # base Qwen3-VL 既不遵守也不受益 —— no-prefill 是本系列的标准协议
    ap.add_argument("--no-prefill", dest="prefill", action="store_false",
                    help="不预填 <think> (Qwen3-VL 系列标准协议)")
    # 32B-Instruct 对含 <think> 标签指令的 system prompt 病态反应 (首 token 即 EOS,
    # 小图+sys 二分实锤); 无 system 时完全正常。<answer> 指令在 USER_SUFFIX 里, 足够。
    ap.add_argument("--no-system", dest="system", action="store_false",
                    help="去掉 system prompt (Qwen3-VL 系列标准协议)")
    ap.add_argument("--no-think", action="store_true",
                    help="关闭模板的 thinking 模式 (Qwen3.5 默认开; 关掉后与 Qwen3-VL-Instruct 可比)")
    args = ap.parse_args()

    tag = args.model.rstrip("/").split("/")[-1].lower()
    variant = args.condition + ("-nothink" if args.no_think else "")
    out_path = args.out or f"{VOPD_ROOT}/eval/model_answer/treebench/{tag}-{variant}_answer.jsonl"

    df = load_dataset("csv", data_files=TSV_PATH, delimiter="\t")["train"]
    if args.limit:
        df = df.select(range(args.limit))
    print(f"loaded {len(df)} rows; condition={args.condition}; out -> {out_path}")

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
    processor = AutoProcessor.from_pretrained(args.model)

    out_f = open(out_path, "a")
    for item in tqdm(df, desc=f"{tag}-{args.condition}"):
        if item["index"] in done:
            continue
        qs = question_text(item)
        img_content = {"type": "image_url", "image_url": f"data:image/jpeg;base64,{item['image']}"}

        stage1_raw, objs = "", []
        if args.condition == "staged":
            s1_messages = [{"role": "user", "content": [
                dict(img_content), {"type": "text", "text": qs + STAGE1_SUFFIX}]}]
            stage1_raw = run_generate(model, processor, s1_messages, args.stage1_max_new_tokens,
                                      no_think=args.no_think)
            objs = parse_stage1_boxes(stage1_raw)

        if args.condition == "boxfirst":
            # 单轮: 先输出原生 grounding JSON, 再推理作答, 一次生成完成
            s2_messages = ([{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_NOBOX}]}]
                           if args.system else []) + [
                {"role": "user", "content": [
                    dict(img_content), {"type": "text", "text": qs + BOXFIRST_SUFFIX}]},
            ]
            output_text = run_generate(model, processor, s2_messages, args.max_new_tokens,
                                       prefill="<think>" if args.prefill else "",
                                       no_think=args.no_think)
            objs = parse_stage1_boxes(output_text)
        else:
            box_block = ""
            if objs:
                lines = "\n".join(f"- {o['label']}: [{','.join(str(int(v)) for v in o['bbox_2d'])}]" for o in objs)
                box_block = f"\nRelevant objects located in the image, as bounding boxes [x1,y1,x2,y2]:\n{lines}\n"

            s2_messages = ([{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_NOBOX}]}]
                           if args.system else []) + [
                {"role": "user", "content": [
                    dict(img_content), {"type": "text", "text": qs + box_block + USER_SUFFIX}]},
            ]
            output_text = run_generate(model, processor, s2_messages, args.max_new_tokens,
                                       prefill="<think>" if args.prefill else "",
                                       no_think=args.no_think)

        w, h = Image.open(io.BytesIO(base64.b64decode(item["image"]))).size
        gt_boxes = ast.literal_eval(item["target_instances"])
        pred_boxes = [o["bbox_2d"] for o in objs]
        pred_norm = [[b[0] * w / 1000, b[1] * h / 1000, b[2] * w / 1000, b[3] * h / 1000] for b in pred_boxes]

        m = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
        ans = m.group(1).strip().upper() if m else output_text

        rec = {
            "index": item["index"], "category": item["category"], "answer": item["answer"],
            "prediction": ans,
            "iou_abs": avg_best_iou(pred_boxes, gt_boxes),
            "iou_norm": avg_best_iou(pred_norm, gt_boxes),
            "pred_boxes": pred_boxes, "labels": [o["label"] for o in objs],
            "image_size": [w, h], "model": args.model, "condition": args.condition,
            "prefill": args.prefill, "system": args.system,
            "stage1_raw": stage1_raw, "output": output_text,
        }
        done[item["index"]] = rec
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()

    summarize(list(done.values()), args.coord_space, args.condition)


if __name__ == "__main__":
    main()
