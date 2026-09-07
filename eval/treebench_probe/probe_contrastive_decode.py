"""瓶颈对比解码探针: "反向 OPD" 想法的免训练存在性检验 (VCD 思想 + 我们的结构化剥夺)。

想法: 信息剥夺视图会诱导模型自信断言看不见的内容(幻觉, dim 实验实锤"质量盲")。
推理时做对比:
    fused = (1+a) * logits(清洁图) - a * logits(剥夺图)
即压低"剥夺视图下反而高概率"的 token(瓶颈特有断言), 保留清洁视图支持的 token。
配 VCD 式 plausibility 约束: 候选集限于 p_clean >= beta * max(p_clean), 防流畅性退化。

若对比解码在 TreeBench 上涨分 => 负向信号存在且可提取, 才值得做训练版(反向 OPD);
不涨则一天内止损。

剥夺视图两档:
  shuffled  (idx+37)%405 那题的 GT 框换算到本图做 hide 融合 —— GT 锚定 (训练时可用)
  randmask  随机框 hide 融合(按题号种子确定性) —— 零 GT, 可部署版

运行 (vision-opd 环境, GPU; 每题两路增量解码, 9B 全量约 2.5-3h, 建议先 --limit 50):
  python probe_contrastive_decode.py --model Qwen/Qwen3.5-9B --alpha 1.0 --deprived shuffled --limit 10
"""
import argparse
import ast
import base64
import io
import json
import os
import random
import re

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from infer_privilege import (VOPD_ROOT, TSV_PATH, USER_SUFFIX, TAGS, question_text,
                             hide_compose, inverse_hide_compose, to_b64)

ANS_DIR = f"{VOPD_ROOT}/eval/model_answer/treebench"


def deprived_image(item, mode, all_gt, all_sizes):
    pil = Image.open(io.BytesIO(base64.b64decode(item["image"])))
    w, h = pil.size
    idx = item["index"]
    if mode == "inverse":
        return to_b64(inverse_hide_compose(pil, all_gt[idx]))
    if mode == "shuffled":
        src = (idx + 37) % 405
        sw, sh = all_sizes[src]
        boxes = [[b[0]*w/sw, b[1]*h/sh, b[2]*w/sw, b[3]*h/sh] for b in all_gt[src]]
    else:  # randmask: 零 GT 依赖, 框数固定 2, 尺寸 20-45% 边长
        rng = random.Random(7000 + idx)
        boxes = []
        for _ in range(2):
            bw, bh = int(w*rng.uniform(.2, .45)), int(h*rng.uniform(.2, .45))
            x1, y1 = rng.randint(0, max(1, w-bw)), rng.randint(0, max(1, h-bh))
            boxes.append([x1, y1, x1+bw, y1+bh])
    return to_b64(hide_compose(pil, boxes, with_box=False))


def prep(processor, model, item, image_b64):
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64 or item['image']}"},
        {"type": "text", "text": question_text(item) + USER_SUFFIX},
    ]}]
    image_inputs, _ = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    inputs = processor(text=[text], images=image_inputs, padding=True,
                       return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs, use_cache=True)
    return out.past_key_values, out.logits[0, -1].float(), inputs["input_ids"].shape[1]


def step(model, pkv, tok, cur_len):
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([[tok]], device=model.device),
                    past_key_values=pkv, use_cache=True,
                    cache_position=torch.tensor([cur_len], device=model.device))
    return out.past_key_values, out.logits[0, -1].float()


def summarize(rows):
    tot = cor = 0
    for t in TAGS:
        g = [x for x in rows if x["category"] == t]
        c = sum(1 for x in g if x["prediction"].upper() == x["answer"].upper())
        tot += len(g); cor += c
        if g:
            print(t, f"{c}/{len(g)}={round(c/len(g)*100, 2)}")
    print(f"==> Overall {cor}/{tot}={round(cor/tot*100, 2)}")
    iv = [r["intervene_rate"] for r in rows]
    print(f"==> 干预率(融合改写了清洁argmax的步数占比): 均值 {sum(iv)/len(iv)*100:.1f}%")
    nf = sum(1 for x in rows if not re.search(r"<answer>.*?</answer>", x["output"], re.DOTALL))
    print(f"==> 无 <answer>: {nf}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1, help="plausibility 约束阈 (VCD 式)")
    ap.add_argument("--deprived", choices=["shuffled", "randmask", "inverse"], default="shuffled")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    tag = args.model.rstrip("/").split("/")[-1].lower()
    out_path = args.out or (f"{ANS_DIR}/{tag}-cdecode-{args.deprived}"
                            f"-a{args.alpha}-nothink_answer.jsonl")
    print(f"out -> {out_path}")

    all_gt, all_sizes = {}, {}
    import csv, sys as _sys
    csv.field_size_limit(_sys.maxsize)
    for row in csv.DictReader(open(TSV_PATH), delimiter="\t"):
        all_gt[int(row["index"])] = ast.literal_eval(row["target_instances"])
    for r in map(json.loads, open(f"{ANS_DIR}/qwen3.5-4b-priv-draw-labeled-nothink_answer.jsonl")):
        all_sizes[r["index"]] = r["image_size"]

    df = load_dataset("csv", data_files=TSV_PATH, delimiter="\t")["train"]
    if args.limit:
        df = df.select(range(args.limit))

    done = {}
    if os.path.exists(out_path):
        done = {r["index"]: r for r in map(json.loads, open(out_path))}
        print(f"resume: {len(done)} rows")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])

    out_f = open(out_path, "a")
    for item in tqdm(df, desc=f"{tag}-cdec-a{args.alpha}"):
        if item["index"] in done:
            continue
        dep_b64 = deprived_image(item, args.deprived, all_gt, all_sizes)
        pkv_c, log_c, len_c = prep(processor, model, item, None)
        pkv_d, log_d, len_d = prep(processor, model, item, dep_b64)

        toks, interv = [], 0
        for _ in range(args.max_new_tokens):
            pc = torch.softmax(log_c, dim=-1)
            plaus = pc >= args.beta * pc.max()                     # 候选集来自清洁分布
            fused = (1 + args.alpha) * log_c - args.alpha * log_d
            fused = fused.masked_fill(~plaus, float("-inf"))
            t = int(fused.argmax())
            if t != int(log_c.argmax()):
                interv += 1
            if t in eos_ids:
                break
            toks.append(t)
            pkv_c, log_c = step(model, pkv_c, t, len_c + len(toks) - 1)
            pkv_d, log_d = step(model, pkv_d, t, len_d + len(toks) - 1)

        output = tokenizer.decode(toks, skip_special_tokens=False)
        m = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        pred = m.group(1).strip().upper() if m else output

        rec = {"index": item["index"], "category": item["category"], "answer": item["answer"],
               "prediction": pred, "alpha": args.alpha, "deprived": args.deprived,
               "intervene_rate": round(interv / max(1, len(toks)), 4),
               "n_tokens": len(toks), "model": args.model, "output": output}
        done[item["index"]] = rec
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    summarize(list(done.values()))


if __name__ == "__main__":
    main()
