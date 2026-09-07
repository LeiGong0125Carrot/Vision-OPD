"""State-Adaptive OPD-Aha (09_01_plan) 的训练前组件验证探针。

注: e_t = D(H,F) − λD(H,S) 在训练起点恒为 0 (student≡frozen teacher, p^S≡p^F),
故本探针验证的是它的两个组件, 而非 e_t 本身:

  V1 干预效应剖面: 沿失败链逐位置算 D_JS(p^H, p^F)(top-100 并集 + tail bucket,
     与 plan 的分布支撑一致) —— 干预表达在链条哪里?预期前置(与 KL 前移一致),
     决定 re-entry gate 将主要开在何处。
  V2 sharpening 前提: 在 ρ=0 答案槽上, s = H(p^F) − H(p^H) 的符号是否与
     残差方向(margin_hide − margin_raw, 指向正确为正)相关 —— s_t 作为软预算的
     合法性检验;并给出各类别的 s 分布(预期 Ordering/上下文删除型 s<0 居多)。

实现: 每题 2 次全链 teacher-forcing forward(raw/hide)拿逐位置分布 +
2 次答案槽 forward。185 条失败链约 12-15 分钟。

运行:
  python probe_state_adaptive.py --model Qwen/Qwen3.5-9B --limit 10   # 冒烟
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
                             hide_compose, to_b64)

ANS_DIR = f"{VOPD_ROOT}/eval/model_answer/treebench"
MAX_POS = 128
TOPK = 100
BUCKETS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128)]
ANSWER_PROMPT = "\nThe correct option is"


def js_topk(logits_a, logits_b, k=TOPK):
    """top-k 并集 + tail bucket 上的 JS 散度 (与 plan 的分布支撑一致)。[T,V] -> [T]"""
    pa = torch.softmax(logits_a, -1)
    pb = torch.softmax(logits_b, -1)
    ia = pa.topk(k, -1).indices
    ib = pb.topk(k, -1).indices
    out = []
    for t in range(pa.shape[0]):
        idx = torch.unique(torch.cat([ia[t], ib[t]]))
        a = pa[t, idx]; b = pb[t, idx]
        a = torch.cat([a, (1 - a.sum()).clamp_min(1e-9).unsqueeze(0)])
        b = torch.cat([b, (1 - b.sum()).clamp_min(1e-9).unsqueeze(0)])
        a = a / a.sum(); b = b / b.sum()
        m = 0.5 * (a + b)
        js = 0.5 * (a * (a / m).log()).sum() + 0.5 * (b * (b / m).log()).sum()
        out.append(float(js))
    return out


def build_inputs(processor, model, item, image_b64, extra_text=""):
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64 or item['image']}"},
        {"type": "text", "text": question_text(item) + USER_SUFFIX},
    ]}]
    image_inputs, _ = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) + extra_text
    return processor(text=[text], images=image_inputs, padding=True,
                     return_tensors="pt").to(model.device), image_inputs


def forced_logits(model, inputs, cont_ids):
    """整链 teacher-forcing, 返回预测各续写 token 的 logits [T,V]。"""
    full_ids = torch.cat([inputs["input_ids"], cont_ids.unsqueeze(0).to(model.device)], dim=1)
    kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    if "mm_token_type_ids" in kwargs:
        pad = torch.zeros((1, cont_ids.shape[0]), dtype=kwargs["mm_token_type_ids"].dtype,
                          device=kwargs["mm_token_type_ids"].device)
        kwargs["mm_token_type_ids"] = torch.cat([kwargs["mm_token_type_ids"], pad], dim=1)
    with torch.inference_mode():
        out = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), **kwargs)
    p0 = inputs["input_ids"].shape[1]
    return out.logits[0, p0 - 1: p0 - 1 + cont_ids.shape[0]].float()


def option_letters(item):
    if item["category"] == "OCR":
        return []
    return re.findall(r"^\s*([A-Z])[\.\)]", str(item["multi-choice options"]), re.M)


def summarize(rows):
    import statistics
    print(f"\n===== state-adaptive 组件验证: {len(rows)} 条失败链")
    print("V1 干预效应 D_JS(H,F) 位置剖面:")
    for lo, hi in BUCKETS:
        v = [x for r in rows for x in r["js"][lo:hi]]
        if v:
            print(f"  [{lo:>3}-{hi:<3}) mean={statistics.mean(v):.4f}  p90={sorted(v)[int(len(v)*0.9)]:.4f}")
    print("\nV2 sharpening 前提 (ρ=0 答案槽):")
    s = [r["slot_s"] for r in rows]
    dm = [r["slot_dmargin"] for r in rows]
    pos = [d for x, d in zip(s, dm) if x > 0]
    neg = [d for x, d in zip(s, dm) if x <= 0]
    print(f"  s>0 的状态 {len(pos)} 个: 残差方向均值 {statistics.mean(pos):+.3f}" if pos else "  s>0: 无")
    print(f"  s<=0 的状态 {len(neg)} 个: 残差方向均值 {statistics.mean(neg):+.3f}" if neg else "  s<=0: 无")
    n = len(s)
    if n > 2:
        ms, md = statistics.mean(s), statistics.mean(dm)
        cov = sum((a - ms) * (b - md) for a, b in zip(s, dm)) / n
        import math
        corr = cov / (math.sqrt(sum((a - ms) ** 2 for a in s) / n) * math.sqrt(sum((b - md) ** 2 for b in dm) / n) + 1e-9)
        print(f"  corr(s, Δmargin) = {corr:.3f}")
    from collections import defaultdict
    byc = defaultdict(list)
    for r in rows:
        byc[r["category"].split("/")[-1]].append(r["slot_s"])
    print("  各类别 slot s 均值:", {c: round(statistics.mean(v), 3) for c, v in byc.items() if len(v) >= 8})


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
    out_path = args.out or f"{ANS_DIR}/{tag}-stateadapt.jsonl"
    print(f"out -> {out_path}")

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
    for item in tqdm(df, desc=f"{tag}-stateadapt"):
        idx = item["index"]
        if idx in done or idx not in trajs:
            continue
        traj = trajs[idx]
        correct = traj["answer"].upper()
        realized = traj["prediction"].upper()[:1]
        letters = option_letters(item)
        if correct == traj["prediction"].upper() or realized not in letters or correct not in letters:
            continue
        cand = {}
        ok = True
        for L in letters:
            ids = tokenizer(" " + L, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                ok = False
                break
            cand[L] = ids[0]
        if not ok:
            continue

        output = traj["output"].split("<|im_end|>")[0]
        cont_ids = torch.tensor(tokenizer(output, add_special_tokens=False)["input_ids"][:MAX_POS])
        pil = Image.open(io.BytesIO(base64.b64decode(item["image"])))
        hide_b64 = to_b64(hide_compose(pil, ast.literal_eval(item["target_instances"])))

        # V1: 全链双条件 forcing -> 逐位置 JS
        in_raw, _ = build_inputs(processor, model, item, None)
        in_hid, _ = build_inputs(processor, model, item, hide_b64)
        lg_raw = forced_logits(model, in_raw, cont_ids)
        lg_hid = forced_logits(model, in_hid, cont_ids)
        js = js_topk(lg_hid, lg_raw)

        # V2: ρ=0 答案槽 (question + answer prompt), 双条件
        slot = {}
        for name, b64 in (("raw", None), ("hide", hide_b64)):
            inp, _ = build_inputs(processor, model, item, b64, extra_text=ANSWER_PROMPT)
            with torch.inference_mode():
                out = model(**inp)
            lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            p = lp.exp()
            slot[name] = {"margin": float(lp[cand[correct]] - lp[cand[realized]]),
                          "H": float(-(p * (p + 1e-12).log()).sum())}
        rec = {"index": idx, "category": item["category"],
               "js": [round(v, 5) for v in js],
               "slot_s": round(slot["raw"]["H"] - slot["hide"]["H"], 4),
               "slot_dmargin": round(slot["hide"]["margin"] - slot["raw"]["margin"], 4)}
        done[idx] = rec
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    summarize(list(done.values()))


if __name__ == "__main__":
    main()
