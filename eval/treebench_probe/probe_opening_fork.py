"""开场分叉干预探针 (F0 实验, 08_31_plan P2 的第一枪)。

背景: init-confidence 探针显示中位首分叉就在 token 0, 且类内残留"开场句式×对错"
相关(4B Persp 'From the' 9% vs 'To determine' 23%; 9B Contact 'Based on' 82% vs
'Looking at' 36%)。本实验把相关升级为因果:

  对每题, 取 direct(nothink) 首 token 分布的 top-2 备选, 强制它为开场, 贪心续写全链,
  看答案是否翻转。

读数:
  错链: P(翻对 | 换开场)  —— 显著高于随机 => token-0 初始化 causally 决定 reasoning basin
  对链: P(翻错 | 换开场)  —— 对链稳健 + 错链易翻 = "错链停在浅盆地" 的证据
  p2 门槛分层: top-2 概率太低的备选属于分布外强灌, 单独统计。

运行 (vision-opd 环境, GPU):
  python probe_opening_fork.py --model Qwen/Qwen3.5-9B --limit 10   # 冒烟
  python probe_opening_fork.py --model Qwen/Qwen3.5-9B              # 全量 ~40min
可选: --force-text "Based on" 把开场强制为指定词组(类内"好开场"移植实验)。
"""
import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

import ast
import base64 as b64mod
import io as iomod
from PIL import Image as PILImage
from infer_privilege import (VOPD_ROOT, TSV_PATH, USER_SUFFIX, TAGS, question_text,
                             hide_compose, to_b64)

ANS_DIR = f"{VOPD_ROOT}/eval/model_answer/treebench"


def build_prompt(item, image_b64=None):
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64 or item['image']}"},
        {"type": "text", "text": question_text(item) + USER_SUFFIX},
    ]}]


def make_priv_image(item, privilege, all_gt, all_sizes):
    """分解实验的图像条件 (文本 prompt 逐字节不变, 纯视觉干预):
    hide          GT 区域黑底原位融合 (冠军形态, de-lock + evidence)
    hide_shuffled 用 (index+37)%405 那题的 GT 框 (按尺寸比例换算到本图) 做同样融合 ——
                  真实框统计、错误位置 => 纯 de-lock 对照"""
    idx = item["index"]
    pil = PILImage.open(iomod.BytesIO(b64mod.b64decode(item["image"])))
    if privilege == "hide":
        boxes = all_gt[idx]
    else:
        src_idx = (idx + 37) % 405
        w, h = pil.size
        sw, sh = all_sizes[src_idx]
        boxes = [[b[0]*w/sw, b[1]*h/sh, b[2]*w/sw, b[3]*h/sh] for b in all_gt[src_idx]]
    return to_b64(hide_compose(pil, boxes))


def pad_mm(kwargs, n, device):
    if "mm_token_type_ids" in kwargs and n > 0:
        pad = torch.zeros((1, n), dtype=kwargs["mm_token_type_ids"].dtype, device=device)
        kwargs["mm_token_type_ids"] = torch.cat([kwargs["mm_token_type_ids"], pad], dim=1)
    return kwargs


def summarize(rows):
    ok = [r for r in rows if r["orig_correct"]]
    ng = [r for r in rows if not r["orig_correct"]]
    print(f"\n===== 开场分叉干预: 原对 {len(ok)} / 原错 {len(ng)}")

    def stats(rs, name):
        if not rs:
            return
        ch = sum(1 for r in rs if r["new_pred"].upper() != r["orig_pred"].upper())
        nc = sum(1 for r in rs if r["new_correct"])
        print(f"[{name}] 预测改变 {ch}/{len(rs)}={ch/len(rs)*100:.1f}%   干预后正确 {nc}/{len(rs)}={nc/len(rs)*100:.1f}%")

    stats(ng, "原错链")
    fixed = sum(1 for r in ng if r["new_correct"])
    print(f"  错→对(救回) {fixed}/{len(ng)}={fixed/len(ng)*100:.1f}%")
    if not ok:
        from collections import Counter
        print("  强制开场 token 分布:", dict(Counter(r["alt_token"].strip() for r in rows).most_common(8)))
        return
    stats(ok, "原对链")
    broke = sum(1 for r in ok if not r["new_correct"])
    print(f"  对→错(打坏) {broke}/{len(ok)}={broke/len(ok)*100:.1f}%")
    # p2 分层 (top-2 概率高 = 真实合理分叉; 低 = 分布外强灌)
    for lo, hi in [(0.05, 1.0), (0.01, 0.05), (0.0, 0.01)]:
        sub = [r for r in ng if lo <= r["alt_p"] < hi]
        if sub:
            f = sum(1 for r in sub if r["new_correct"])
            print(f"  错链@alt_p∈[{lo},{hi}): 救回 {f}/{len(sub)}={f/len(sub)*100:.1f}%")
    # 按类别的错链救回
    print("  错链按类救回:")
    for t in TAGS:
        g = [r for r in ng if r["category"] == t]
        if len(g) >= 5:
            f = sum(1 for r in g if r["new_correct"])
            print(f"    {t.split('/')[-1]:<24} {f}/{len(g)}")
    from collections import Counter
    print("  强制开场 token 分布:", dict(Counter(r["alt_token"].strip() for r in rows).most_common(8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--traj-from", default=None)
    ap.add_argument("--force-text", default=None,
                    help="强制固定开场词组(如 'Based on'); 默认用各题自己的 top-2 token")
    ap.add_argument("--privilege", choices=["none", "hide", "hide_shuffled"], default="none",
                    help="分解实验图像条件; hide/hide_shuffled 时复用 top2 基线 run 的强制开场")
    ap.add_argument("--subset", choices=["all", "wrong"], default="all",
                    help="wrong = 只跑原错链 (分解实验默认关注人群)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    tag = args.model.rstrip("/").split("/")[-1].lower()
    traj_path = args.traj_from or f"{ANS_DIR}/{tag}-direct-nothink_answer.jsonl"
    trajs = {r["index"]: r for r in map(json.loads, open(traj_path))}
    variant = ("forcedtext" if args.force_text else "top2") + (
        "" if args.privilege == "none" else f"-{args.privilege}")
    out_path = args.out or f"{ANS_DIR}/{tag}-openfork-{variant}_answer.jsonl"

    # 特权条件下: 复用 raw 基线 run 的强制开场 token (同一分叉、只换视觉上下文)
    base_alt = {}
    if args.privilege != "none":
        base_path = f"{ANS_DIR}/{tag}-openfork-top2_answer.jsonl"
        base_alt = {r["index"]: r for r in map(json.loads, open(base_path))}
        print(f"复用基线分叉 token: {base_path} ({len(base_alt)})")
    all_gt, all_sizes = {}, {}
    if args.privilege != "none":
        import csv, sys as _sys
        csv.field_size_limit(_sys.maxsize)
        for row in csv.DictReader(open(TSV_PATH), delimiter="\t"):
            all_gt[int(row["index"])] = ast.literal_eval(row["target_instances"])
        for r in map(json.loads, open(f"{ANS_DIR}/qwen3.5-4b-priv-draw-labeled-nothink_answer.jsonl")):
            all_sizes[r["index"]] = r["image_size"]
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
    forced_ids = (tokenizer(args.force_text, add_special_tokens=False)["input_ids"]
                  if args.force_text else None)

    out_f = open(out_path, "a")
    for item in tqdm(df, desc=f"{tag}-openfork"):
        if item["index"] in done or item["index"] not in trajs:
            continue
        traj = trajs[item["index"]]
        if args.subset == "wrong" and traj["prediction"].upper() == item["answer"].upper():
            continue

        priv_b64 = (make_priv_image(item, args.privilege, all_gt, all_sizes)
                    if args.privilege != "none" else None)
        messages = build_prompt(item, priv_b64)
        image_inputs, _ = process_vision_info(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        inputs = processor(text=[text], images=image_inputs, padding=True,
                           return_tensors="pt").to(model.device)

        # 首 token 分布 -> 备选开场
        with torch.inference_mode():
            out0 = model(**inputs)
        probs = torch.softmax(out0.logits[0, -1].float(), dim=-1)
        top = probs.topk(5)
        if args.privilege != "none" and item["index"] in base_alt:
            alt_tok = base_alt[item["index"]]["alt_token"]
            ids = tokenizer(alt_tok, add_special_tokens=False)["input_ids"]
            alt = torch.tensor(ids, device=model.device)
            alt_p = float(probs[ids[0]])
        elif forced_ids is not None:
            alt = torch.tensor(forced_ids, device=model.device)
            alt_p = float(probs[forced_ids[0]])
            alt_tok = args.force_text
        else:
            # top-2; 若 top-2 与 top-1 解码后仅大小写/空白差异, 顺延取下一个
            t1 = tokenizer.decode([int(top.indices[0])])
            k = 1
            while k < 5 and tokenizer.decode([int(top.indices[k])]).strip().lower() == t1.strip().lower():
                k += 1
            alt = top.indices[k:k+1]
            alt_p = float(top.values[k])
            alt_tok = tokenizer.decode([int(alt[0])])

        full_ids = torch.cat([inputs["input_ids"], alt.unsqueeze(0)], dim=1)
        kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        kwargs = pad_mm(kwargs, full_ids.shape[1] - inputs["input_ids"].shape[1], model.device)
        with torch.inference_mode():
            gen = model.generate(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
                                 **kwargs, top_p=0.001, top_k=1, temperature=0.01,
                                 repetition_penalty=1.0, max_new_tokens=args.max_new_tokens,
                                 use_cache=True, do_sample=True)
        new_out = alt_tok + processor.batch_decode(
            [gen[0][full_ids.shape[1]:]], skip_special_tokens=False,
            clean_up_tokenization_spaces=False)[0]
        m = re.search(r"<answer>(.*?)</answer>", new_out, re.DOTALL)
        new_pred = m.group(1).strip().upper() if m else new_out

        rec = {
            "index": item["index"], "category": item["category"], "answer": item["answer"],
            "orig_pred": traj["prediction"],
            "orig_correct": traj["prediction"].upper() == item["answer"].upper(),
            "alt_token": alt_tok, "alt_p": round(alt_p, 4),
            "new_pred": new_pred,
            "new_correct": new_pred.upper() == item["answer"].upper(),
            "model": args.model, "variant": variant, "output": new_out,
        }
        done[item["index"]] = rec
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    summarize(list(done.values()))


if __name__ == "__main__":
    main()
