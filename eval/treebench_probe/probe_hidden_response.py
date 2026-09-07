"""Hidden-response 探针 (仿 OPD-Aha Probe A+B, TreeVGR/probe.md 方法论):
在固定的失败轨迹上做离线 replay, 检验 (1) prefix dominance 是否复现,
(2) hidden visual correction 是否存在, (3) 反向蒸馏的前提 —— inverse 视图
是否相对 null 有"指向错误答案"的方向性危害。

对每条 9B 失败链 (nothink):
  保留错误 reasoning 的前 rho 比例 token + 公共答案提示 "\\nThe correct option is"
  在四个视觉条件下 forward 一次, 读下一 token 位置上各选项字母 (" A"/" B"/...) 的 logp:
    raw      学生视图 (原图)
    hide     teacher+ (黑底原位融合, 冠军配置)
    inverse  teacher- 候选 (互补掩码: 涂黑 GT 区域)
    null     视觉空白 (整图 mean-RGB, 尺寸同原图) —— Probe B 的参照系
  margin = logp(正确字母) - logp(实际错误字母)

读数:
  A. margin_hide(rho) 曲线: prefix dominance 衰减/翻符号 (对标他们 Fig1a)
  B. margin_hide - margin_null: hidden visual correction 随 rho 的存续 (Fig1b)
  C. margin_inverse - margin_null: 反向蒸馏判决 —— 显著为负才有"可减的危害方向"

运行 (vision-opd 环境, GPU; 207 错链 x 5 rho x 4 条件 = ~4140 次 forward, ~40min):
  python probe_hidden_response.py --model Qwen/Qwen3.5-9B --limit 10   # 冒烟
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

from infer_privilege import (VOPD_ROOT, TSV_PATH, USER_SUFFIX, question_text,
                             hide_compose, inverse_hide_compose, to_b64)

ANS_DIR = f"{VOPD_ROOT}/eval/model_answer/treebench"
RHOS = [0.0, 0.25, 0.5, 0.75, 0.9]
ANSWER_PROMPT = "\nThe correct option is"
CONDS = ["raw", "hide", "inverse", "null"]


def null_image(pil):
    import numpy as np
    arr = np.array(pil.convert("RGB"))
    mean = tuple(int(v) for v in arr.reshape(-1, 3).mean(0))
    return Image.new("RGB", pil.size, mean)


def option_letters(item):
    if item["category"] == "OCR":
        return []
    return re.findall(r"^\s*([A-Z])[\.\)]", str(item["multi-choice options"]), re.M)


def reasoning_part(output):
    o = output.split("<|im_end|>")[0]
    p = o.find("<answer>")
    return (o[:p] if p >= 0 else o).rstrip()


def summarize(rows):
    import statistics
    print(f"\n===== hidden-response: {len(rows)} 条失败链")
    print(f"{'rho':>5} | " + " | ".join(f"{c:>16}" for c in CONDS) + " |  hid-null  inv-null")
    for ri, rho in enumerate(RHOS):
        line = f"{rho:>5} | "
        margins = {}
        for c in CONDS:
            v = [r["margin"][c][ri] for r in rows if r["margin"][c][ri] is not None]
            rec = [r["rec"][c][ri] for r in rows if r["rec"][c][ri] is not None]
            margins[c] = statistics.mean(v)
            line += f"m{statistics.mean(v):+6.2f} rec{statistics.mean(rec)*100:3.0f}% | "
        line += f"  {margins['hide']-margins['null']:+6.2f}   {margins['inverse']-margins['null']:+6.2f}"
        print(line)
    print("(m = 平均margin[logp正确-logp错误], rec = 正确字母为argmax的比例)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--traj-from", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tag = args.model.rstrip("/").split("/")[-1].lower()
    traj_path = args.traj_from or f"{ANS_DIR}/{tag}-direct-nothink_answer.jsonl"
    trajs = {r["index"]: r for r in map(json.loads, open(traj_path))}
    out_path = args.out or f"{ANS_DIR}/{tag}-hiddenresp.jsonl"
    print(f"trajectories: {traj_path}\nout -> {out_path}")

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

    out_f = open(out_path, "a")
    for item in tqdm(df, desc=f"{tag}-hiddenresp"):
        idx = item["index"]
        if idx in done or idx not in trajs:
            continue
        traj = trajs[idx]
        correct = traj["answer"].upper()
        realized = traj["prediction"].upper()[:1]
        letters = option_letters(item)
        # 只取失败链且实现答案是有效选项字母
        if correct == traj["prediction"].upper() or realized not in letters or correct not in letters:
            continue
        # 候选字母 token (" A" 形式, 单 token)
        cand = {}
        skip = False
        for L in letters:
            ids = tokenizer(" " + L, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                skip = True
                break
            cand[L] = ids[0]
        if skip:
            continue

        reasoning = reasoning_part(traj["output"])
        r_ids = tokenizer(reasoning, add_special_tokens=False)["input_ids"]
        pil = Image.open(io.BytesIO(base64.b64decode(item["image"])))
        gt = ast.literal_eval(item["target_instances"])
        images = {"raw": None,
                  "hide": to_b64(hide_compose(pil, gt)),
                  "inverse": to_b64(inverse_hide_compose(pil, gt)),
                  "null": to_b64(null_image(pil))}

        rec = {"index": idx, "category": item["category"], "answer": correct,
               "realized": realized,
               "margin": {c: [] for c in CONDS}, "rec": {c: [] for c in CONDS}}
        for c in CONDS:
            messages = [{"role": "user", "content": [
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{images[c] or item['image']}"},
                {"type": "text", "text": question_text(item) + USER_SUFFIX},
            ]}]
            image_inputs, _ = process_vision_info(messages)
            base_text = processor.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True,
                                                      enable_thinking=False)
            for rho in RHOS:
                prefix = tokenizer.decode(r_ids[:int(len(r_ids) * rho)]) if rho > 0 else ""
                text = base_text + prefix + ANSWER_PROMPT
                inputs = processor(text=[text], images=image_inputs, padding=True,
                                   return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    out = model(**inputs)
                lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                lps = {L: float(lp[t]) for L, t in cand.items()}
                rec["margin"][c].append(round(lps[correct] - lps[realized], 4))
                rec["rec"][c].append(1 if max(lps, key=lps.get) == correct else 0)
        done[idx] = rec
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    summarize([r for r in done.values()])


if __name__ == "__main__":
    main()
