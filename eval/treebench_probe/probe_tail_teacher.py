"""学生尾部 token 的教师意见分布: 盲压制 (OPSA) vs 教师选择子集的分歧率。

对 step-1 rollout (T=1.0, 生成策略=base=F视图教师), 每条链做 F/H 双 forcing:
  学生 logp = lr_F (step1 时 p^S≡p^F);  教师意见 u = lr_H − lr_F。
对学生尾部 (链内 lowest-20% logp; 另报 p<0.1 / p<0.01 子集):
  gems    u > +1   证据教师想救的 token  -> OPSA 会错杀
  neutral |u|<= 1  教师无强意见
  confirm u < -1   共识垃圾              -> 两方法一致
对照: 非尾部 token 的同款分布。
附: 尾部高频 token 类型 + 各自 mean u (修正词是否在尾部且被教师保护)。

运行: python probe_tail_teacher.py            # 768 x 2 forwards, ~6 min
"""
import argparse
import json
import os
import re
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from infer_privilege import VOPD_ROOT

PARQUET = f"{VOPD_ROOT}/data/TreeVGR-RL-37K/train_sa4k.parquet"
DUMP = f"{VOPD_ROOT}/rollouts/SA-OPD-Qwen3.5-4B/1.jsonl"
OUT = f"{VOPD_ROOT}/eval/treebench_probe/tail_teacher_rows.jsonl"
MAX_POS = 384


def build_inputs(processor, model, contents):
    messages = [{"role": "user", "content": contents}]
    image_inputs, _ = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    return processor(text=[text], images=image_inputs, padding=True,
                     return_tensors="pt").to(model.device)


def realized_logp(model, inputs, cont_ids):
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
    return lp.gather(-1, cont_ids.to(lp.device).unsqueeze(-1)).squeeze(-1)


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
            samples.append((li, qmap[q], r["output"]))
        if args.n and len(samples) >= args.n:
            break
    print(f"rollout 链: {len(samples)}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer

    done = set()
    tail_types = defaultdict(lambda: [0.0, 0])   # token id -> [sum_u, n] (仅尾部)
    rows = []
    if os.path.exists(OUT):
        for r in map(json.loads, open(OUT)):
            done.add(r["li"]); rows.append(r)
        print(f"resume: {len(done)}")
    out_f = open(OUT, "a")
    for li, row, output in tqdm(samples, desc="tail-teacher"):
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
        lr_F = realized_logp(model, in_F, cont_ids).cpu()
        lr_H = realized_logp(model, in_H, cont_ids).cpu()
        u = lr_H - lr_F
        T = cont_ids.numel()
        k = max(1, int(0.2 * T))
        tail_idx = lr_F.topk(k, largest=False).indices
        tail_mask = torch.zeros(T, dtype=torch.bool); tail_mask[tail_idx] = True

        def bucket(mask):
            uu = u[mask]
            n = int(mask.sum())
            if n == 0:
                return dict(n=0, gems=0.0, neutral=0.0, confirm=0.0, mean_u=0.0)
            return dict(n=n, gems=float((uu > 1).float().mean()),
                        neutral=float(((uu >= -1) & (uu <= 1)).float().mean()),
                        confirm=float((uu < -1).float().mean()),
                        mean_u=float(uu.mean()))
        p = lr_F.exp()
        rec = dict(li=li, task=str(row["extra_info"]["task"]),
                   tail=bucket(tail_mask), head=bucket(~tail_mask),
                   p10=bucket(p < 0.1), p01=bucket(p < 0.01))
        for t in tail_idx.tolist():
            v = int(cont_ids[t]); tail_types[v][0] += float(u[t]); tail_types[v][1] += 1
        rows.append(rec)
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
    out_f.close()

    import statistics as st
    print(f"\n===== 学生尾部的教师意见: {len(rows)} 条链")
    for key, lbl in [("tail", "lowest-20% (OPSA人群)"), ("p10", "p<0.1"), ("p01", "p<0.01"), ("head", "非尾部对照")]:
        subs = [r[key] for r in rows if r[key]["n"] > 0]
        ntok = sum(s["n"] for s in subs)
        w = lambda f: sum(s[f] * s["n"] for s in subs) / max(ntok, 1)
        print(f"  {lbl:<22} tokens={ntok:>7}  gems(u>+1) {w('gems')*100:5.1f}%  "
              f"neutral {w('neutral')*100:5.1f}%  confirm(u<-1) {w('confirm')*100:5.1f}%  mean_u {w('mean_u'):+.3f}")
    if tail_types:
        print("\n  尾部高频 token 类型 (n>=40) 按 mean_u 排序 — 教师想救谁/想埋谁:")
        cand = [(v, s / c, c) for v, (s, c) in tail_types.items() if c >= 40]
        cand.sort(key=lambda x: -x[1])
        for v, mu, c in cand[:12]:
            print(f"    救: {tokenizer.decode([v])!r:>14}  mean_u={mu:+.2f}  n={c}")
        for v, mu, c in cand[-12:]:
            print(f"    埋: {tokenizer.decode([v])!r:>14}  mean_u={mu:+.2f}  n={c}")


if __name__ == "__main__":
    main()
