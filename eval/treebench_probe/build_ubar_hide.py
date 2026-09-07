"""构建 hide 口径的 EVT ū 预热表 (region 版 evt_ubar_init.json 的替代)。

对 step-1 rollout 链 (base 策略, 与教师形态无关, 仍有效) 做 F(全图)/H(hide合成图) 双 forcing,
按词聚合 u = lrH − lrF 的均值 (出现>=3 次入表)。
输出: data/TreeVGR-RL-37K/evt_ubar_init_hide.json
附: |ū| 最大词表 —— 验证 hide 语域 (预期无 cropped/provided, 有黑底/暗场词)。

运行 (单卡 ~7min): python build_ubar_hide.py
"""
import json
import re
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from infer_privilege import VOPD_ROOT
from probe_tail_teacher import build_inputs, realized_logp

PARQUET = f"{VOPD_ROOT}/data/TreeVGR-RL-37K/train_sa4k_hide.parquet"
DUMP = f"{VOPD_ROOT}/rollouts/SA-OPD-Qwen3.5-4B/1.jsonl"
OUT = f"{VOPD_ROOT}/data/TreeVGR-RL-37K/evt_ubar_init_hide.json"
MAX_POS = 384


def main():
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
            samples.append((qmap[q], r["output"]))
    print(f"rollout 链: {len(samples)}")

    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3.5-4B", torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B")
    tokenizer = processor.tokenizer

    usum, ucnt = defaultdict(float), defaultdict(int)
    for row, output in tqdm(samples, desc="ubar-hide"):
        cont_ids = torch.tensor(tokenizer(output, add_special_tokens=False)["input_ids"][:MAX_POS])
        if cont_ids.numel() < 10:
            continue
        question = str(row["extra_info"]["question"]).strip()
        in_F = build_inputs(processor, model,
                            [{"type": "image", "image": row["images"][0]["path"]},
                             {"type": "text", "text": question}])
        in_H = build_inputs(processor, model,
                            [{"type": "image", "image": row["bbox_images"][0]["path"]},
                             {"type": "text", "text": question}])
        u = (realized_logp(model, in_H, cont_ids) - realized_logp(model, in_F, cont_ids)).cpu()
        for v, val in zip(cont_ids.tolist(), u.tolist()):
            usum[v] += val; ucnt[v] += 1

    warm = {str(v): round(usum[v] / ucnt[v], 4) for v in usum if ucnt[v] >= 3}
    json.dump(warm, open(OUT, "w"))
    print(f"hide ū 表: {len(warm)} 词 -> {OUT}")
    cand = [(v, usum[v] / ucnt[v], ucnt[v]) for v in usum if ucnt[v] >= 40]
    cand.sort(key=lambda x: -abs(x[1]))
    print("|ū| 最大词 (n>=40):")
    for v, m, c in cand[:16]:
        print(f"  {tokenizer.decode([v])!r:>16}  ū={m:+.2f}  n={c}")


if __name__ == "__main__":
    main()
