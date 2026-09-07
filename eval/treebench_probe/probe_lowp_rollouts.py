"""训练 rollout 版低概率结构统计: OPSA 在我们训练分布上的"弹药量"。

对 step-1 dump 的全部采样链 (T=1.0 采样, 生成策略 = base 4B, 自评分精确成立):
在 full-image 视图下 teacher-forcing, 统计与 probe_lowp_chains 同款指标。
对照物: 贪心评测链 (tail20_logp≈-0.72, frac_p10≈0.006%, top1≈98.5%)。

运行: python probe_lowp_rollouts.py            # 768 条, ~3-5 min
注: 后期 step 的链由漂移后的策略生成, 自评分需要对应 checkpoint, 本探针不覆盖。
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
from probe_lowp_chains import forced_stats

PARQUET = f"{VOPD_ROOT}/data/TreeVGR-RL-37K/train_sa4k.parquet"
DUMP_DIR = f"{VOPD_ROOT}/rollouts/SA-OPD-Qwen3.5-4B"
OUT = f"{VOPD_ROOT}/eval/treebench_probe/lowp_rollouts_rows.jsonl"
MAX_POS = 384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--n", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    qmap = {}
    for _, row in df.iterrows():
        qmap.setdefault(str(row["extra_info"]["question"]).strip(), row)

    samples = []
    for li, line in enumerate(open(f"{DUMP_DIR}/{args.step}.jsonl")):
        r = json.loads(line)
        m = re.search(r"user\n\n?(.*?)\nassistant", r["input"], re.S)
        q = (m.group(1).strip() if m else "")
        if q in qmap and len(r["output"].split()) >= 5:
            samples.append((li, qmap[q], r["output"]))
        if args.n and len(samples) >= args.n:
            break
    print(f"rollout 链: {len(samples)} (step {args.step})")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer

    done = {}
    if os.path.exists(OUT):
        done = {r["li"]: r for r in map(json.loads, open(OUT)) if r.get("step") == args.step}
        print(f"resume: {len(done)}")
    out_f = open(OUT, "a")
    for li, row, output in tqdm(samples, desc=f"rollout-lowp-s{args.step}"):
        if li in done:
            continue
        cont_ids = torch.tensor(tokenizer(output, add_special_tokens=False)["input_ids"][:MAX_POS])
        if cont_ids.numel() < 5:
            continue
        question = str(row["extra_info"]["question"]).strip()
        messages = [{"role": "user", "content": [
            {"type": "image", "image": row["images"][0]["path"]},
            {"type": "text", "text": question},
        ]}]
        image_inputs, _ = process_vision_info(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        inputs = processor(text=[text], images=image_inputs, padding=True,
                           return_tensors="pt").to(model.device)
        stats = forced_stats(model, inputs, cont_ids)
        rec = dict(li=li, step=args.step, orig_row=int(row["extra_info"]["orig_row"]),
                   task=str(row["extra_info"]["task"]), **stats)
        done[li] = rec
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
    out_f.close()

    import statistics as st
    rows = [r for r in done.values()]
    ntok = sum(r["T"] for r in rows)
    print(f"\n===== rollout (T=1.0 采样) 低概率结构: {len(rows)} 条链, {ntok} tokens")
    wavg = lambda k: sum(r[k] * r["T"] for r in rows) / max(ntok, 1)
    print(f"  token 加权占比: p<0.5 {wavg('frac_p50')*100:.2f}%  p<0.1 {wavg('frac_p10')*100:.3f}%  "
          f"p<0.01 {wavg('frac_p01')*100:.3f}%  p<0.001 {wavg('frac_p001')*100:.4f}%")
    print(f"  tail20_logp: 中位 {st.median(r['tail20_logp'] for r in rows):.3f}  "
          f"(贪心评测链参照 ≈ -0.72)")
    print(f"  top1_rate: 中位 {st.median(r['top1_rate'] for r in rows):.3f}  (贪心链参照 0.985)")
    print(f"  熵: 中位 {st.median(r['mean_ent'] for r in rows):.3f}  高熵位(H>1)占比 {wavg('frac_H1')*100:.1f}%")


if __name__ == "__main__":
    main()
