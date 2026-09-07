"""OPSA 仿写探针第一步: 对 step-1 rollout 存逐位置数组 (lr_F=学生logp, lr_H, ent_F)。

供 probe_opsa_analysis.py (CPU, import 官方 compute_opsa) 做 Fig2a/3a/3b 忠实复刻
与官方口径的选择器交叉分析。

运行: python probe_opsa_dump.py            # 768 x 2 forwards + 熵, ~7 min
"""
import argparse
import json
import os
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from infer_privilege import VOPD_ROOT
from probe_tail_teacher import build_inputs

PARQUET = f"{VOPD_ROOT}/data/TreeVGR-RL-37K/train_sa4k.parquet"
DUMP = f"{VOPD_ROOT}/rollouts/SA-OPD-Qwen3.5-4B/1.jsonl"
OUT = f"{VOPD_ROOT}/eval/treebench_probe/opsa_dump_rows.jsonl"
MAX_POS = 384


def forced_lp_ent(model, inputs, cont_ids, want_ent):
    full_ids = torch.cat([inputs["input_ids"], cont_ids.unsqueeze(0).to(model.device)], dim=1)
    kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    if "mm_token_type_ids" in kwargs:
        pad = torch.zeros((1, cont_ids.shape[0]), dtype=kwargs["mm_token_type_ids"].dtype,
                          device=kwargs["mm_token_type_ids"].device)
        kwargs["mm_token_type_ids"] = torch.cat([kwargs["mm_token_type_ids"], pad], dim=1)
    with torch.inference_mode():
        out = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), **kwargs)
    p0 = inputs["input_ids"].shape[1]
    lp = torch.log_softmax(out.logits[0, p0 - 1: p0 - 1 + cont_ids.shape[0]].float(), dim=-1)
    lr = lp.gather(-1, cont_ids.to(lp.device).unsqueeze(-1)).squeeze(-1)
    ent = -(lp.exp() * lp).sum(-1) if want_ent else None
    return lr.cpu(), (ent.cpu() if ent is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    qmap = {}
    for _, row in df.iterrows():
        qmap.setdefault(str(row["extra_info"]["question"]).strip(), row)
    samples = []
    for li, line in enumerate(open(DUMP)):
        r = json.loads(line)
        m = re.search(r"user\n\n?(.*?)\nassistant", r["input"], re.S)
        q = (m.group(1).strip() if m else "")
        if q in qmap and len(r["output"].split()) >= 5:
            samples.append((li, qmap[q], r["output"], r["gts"]))
        if args.n and len(samples) >= args.n:
            break
    print(f"rollout 链: {len(samples)}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer

    done = set()
    if os.path.exists(OUT):
        done = {r["li"] for r in map(json.loads, open(OUT))}
        print(f"resume: {len(done)}")
    out_f = open(OUT, "a")
    for li, row, output, gts in tqdm(samples, desc="opsa-dump"):
        if li in done:
            continue
        cont_ids = torch.tensor(tokenizer(output, add_special_tokens=False)["input_ids"][:MAX_POS])
        if cont_ids.numel() < 10:
            continue
        question = str(row["extra_info"]["question"]).strip()
        in_F = build_inputs(processor, model,
                            [{"type": "image", "image": row["images"][0]["path"]},
                             {"type": "text", "text": question}])
        in_H = build_inputs(processor, model,
                            [*[{"type": "image", "image": b["path"]} for b in row["bbox_images"]],
                             {"type": "text", "text": question}])
        lr_F, ent_F = forced_lp_ent(model, in_F, cont_ids, want_ent=True)
        lr_H, _ = forced_lp_ent(model, in_H, cont_ids, want_ent=False)
        rec = dict(li=li, task=str(row["extra_info"]["task"]), question=question, gts=gts,
                   output=output[:6000],
                   tok=[int(v) for v in cont_ids],
                   lrF=[round(float(v), 4) for v in lr_F],
                   lrH=[round(float(v), 4) for v in lr_H],
                   entF=[round(float(v), 4) for v in ent_F])
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
