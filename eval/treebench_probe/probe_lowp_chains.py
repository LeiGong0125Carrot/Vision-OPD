"""OPSA 视角探针: 对/错推理链的低概率 token 结构, direct vs hide无句 (4B, TreeBench)。

每条链在生成它的视图下自评分 (teacher-forcing), 统计:
  frac_p<0.5 / <0.1 / <0.01   低概率 token 占比 (绝对阈值)
  tail20_logp                 链内 lowest-20% token 的平均 logp (尾部深度, OPSA 口径)
  mean_entropy, frac_H>1      逐位置熵
  top1_rate                   实际 token 是 argmax 的比例 (贪心生成应≈1 -> 错误=自信错误)
按 (视图 x 对错) 四格 + 2x2 翻转组汇总。

运行: python probe_lowp_chains.py            # 405 题 x 2 视图, ~5-8 min
"""
import argparse
import ast
import base64
import io
import json
import os

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from infer_privilege import (VOPD_ROOT, TSV_PATH, USER_SUFFIX, question_text,
                             hide_compose, to_b64)

ANS_DIR = f"{VOPD_ROOT}/eval/model_answer/treebench"
ARMS = {
    "direct": "qwen3.5-4b-direct-nothink_answer.jsonl",
    "hide":   "qwen3.5-4b-priv-hide-nosent-nothink_answer.jsonl",
}
MAX_POS = 384
OUT = f"{VOPD_ROOT}/eval/treebench_probe/lowp_chains_rows.jsonl"


def build_inputs(processor, model, item, image_b64):
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64 or item['image']}"},
        {"type": "text", "text": question_text(item) + USER_SUFFIX},
    ]}]
    image_inputs, _ = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    return processor(text=[text], images=image_inputs, padding=True,
                     return_tensors="pt").to(model.device)


def forced_stats(model, inputs, cont_ids):
    full_ids = torch.cat([inputs["input_ids"], cont_ids.unsqueeze(0).to(model.device)], dim=1)
    kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    if "mm_token_type_ids" in kwargs:
        pad = torch.zeros((1, cont_ids.shape[0]), dtype=kwargs["mm_token_type_ids"].dtype,
                          device=kwargs["mm_token_type_ids"].device)
        kwargs["mm_token_type_ids"] = torch.cat([kwargs["mm_token_type_ids"], pad], dim=1)
    with torch.inference_mode():
        out = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), **kwargs)
    p0 = inputs["input_ids"].shape[1]
    logits = out.logits[0, p0 - 1: p0 - 1 + cont_ids.shape[0]].float()
    lp = torch.log_softmax(logits, -1)
    realized = cont_ids.to(lp.device)
    lr = lp.gather(-1, realized.unsqueeze(-1)).squeeze(-1)          # 实际 token logp [T]
    p = lr.exp()
    ent = -(lp.exp() * lp).sum(-1)                                   # 逐位置熵 [T]
    top1 = (lp.argmax(-1) == realized).float()
    T = cont_ids.shape[0]
    k = max(1, int(0.2 * T))
    tail20 = lr.topk(k, largest=False).values
    return dict(T=T,
                mean_logp=float(lr.mean()),
                frac_p50=float((p < 0.5).float().mean()),
                frac_p10=float((p < 0.1).float().mean()),
                frac_p01=float((p < 0.01).float().mean()),
                frac_p001=float((p < 0.001).float().mean()),
                tail20_logp=float(tail20.mean()),
                mean_ent=float(ent.mean()),
                frac_H1=float((ent > 1.0).float().mean()),
                top1_rate=float(top1.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    answers = {}
    for arm, fn in ARMS.items():
        answers[arm] = {r["index"]: r for r in map(json.loads, open(f"{ANS_DIR}/{fn}"))}

    df = load_dataset("csv", data_files=TSV_PATH, delimiter="\t")["train"]
    if args.limit:
        df = df.select(range(args.limit))

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer

    done = {}
    if os.path.exists(OUT):
        done = {(r["index"], r["arm"]): r for r in map(json.loads, open(OUT))}
        print(f"resume: {len(done)}")
    out_f = open(OUT, "a")
    for item in tqdm(df, desc="lowp"):
        idx = item["index"]
        pil = None
        for arm in ARMS:
            if (idx, arm) in done or idx not in answers[arm]:
                continue
            r = answers[arm][idx]
            output = r["output"].split("<|im_end|>")[0]
            cont_ids = torch.tensor(tokenizer(output, add_special_tokens=False)["input_ids"][:MAX_POS])
            if cont_ids.numel() < 8:
                continue
            if arm == "hide":
                if pil is None:
                    pil = Image.open(io.BytesIO(base64.b64decode(item["image"])))
                b64 = to_b64(hide_compose(pil, ast.literal_eval(item["target_instances"])))
            else:
                b64 = None
            inputs = build_inputs(processor, model, item, b64)
            stats = forced_stats(model, inputs, cont_ids)
            ok = r["prediction"].strip().upper()[:1] == r["answer"].strip().upper()[:1]
            rec = dict(index=idx, arm=arm, correct=bool(ok),
                       category=item["category"].split("/")[-1], **stats)
            done[(idx, arm)] = rec
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
    out_f.close()

    # ---------- 汇总 ----------
    import statistics as st
    rows = list(done.values())
    print(f"\n===== 低概率结构: {len(rows)} 条链")
    KEYS = ["frac_p50", "frac_p10", "frac_p01", "tail20_logp", "mean_ent", "frac_H1", "top1_rate", "mean_logp"]
    print(f"{'格':<14}{'n':>5}" + "".join(f"{k:>12}" for k in KEYS))
    for arm in ARMS:
        for ok in (True, False):
            sub = [r for r in rows if r["arm"] == arm and r["correct"] == ok]
            if not sub:
                continue
            tag = f"{arm}-{'对' if ok else '错'}"
            print(f"{tag:<14}{len(sub):>5}" + "".join(f"{st.mean([r[k] for r in sub]):>12.3f}" for k in KEYS))
    # 2x2 配对: 同题双臂
    byidx = {}
    for r in rows:
        byidx.setdefault(r["index"], {})[r["arm"]] = r
    print("\n配对 2x2 (同题, direct/hide 都有):")
    groups = {"A_both_ok": [], "B_hide_rescue": [], "C_hide_harm": [], "D_both_wrong": []}
    for i, d in byidx.items():
        if "direct" not in d or "hide" not in d:
            continue
        dc, hc = d["direct"]["correct"], d["hide"]["correct"]
        g = "A_both_ok" if dc and hc else "B_hide_rescue" if hc else "C_hide_harm" if dc else "D_both_wrong"
        groups[g].append((d["direct"], d["hide"]))
    for g, pairs in groups.items():
        if not pairs:
            continue
        f10d = st.mean([a["frac_p10"] for a, b in pairs])
        f10h = st.mean([b["frac_p10"] for a, b in pairs])
        entd = st.mean([a["mean_ent"] for a, b in pairs])
        enth = st.mean([b["mean_ent"] for a, b in pairs])
        print(f"  {g:<14} n={len(pairs):>3}  frac_p10: direct {f10d:.3f} -> hide {f10h:.3f}   "
              f"熵: {entd:.3f} -> {enth:.3f}")


if __name__ == "__main__":
    main()
