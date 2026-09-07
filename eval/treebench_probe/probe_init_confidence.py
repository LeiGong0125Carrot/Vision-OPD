"""初始化置信度探针: 验证 08_31_plan §18 的 premature-commitment 假设的第一步。

问题: 错误链的早期 token 是否表现出与正确链相当甚至更高的置信度
(即错误源于"过早高置信的轨迹初始化", 而不是"不确定下的犹豫")?

方法: 对已有 direct(nothink) 轨迹逐 token teacher-forcing(纯 forward, 不生成),
记录每条链前 MAX_POS 个位置的:
  p1(top-1 概率) / margin(p1-p2) / entropy(nats) / top5 累积质量 / 实际 token 的 rank
然后按 链最终对/错 分群, 按位置分桶对比。

nothink 模板 (enable_thinking=False) 与轨迹生成时逐字节同构; 复用 probe_logprob 的
forward 拼接机制 (mm_token_type_ids 补零)。

运行 (vision-opd 环境, 需 GPU):
  python probe_init_confidence.py --model Qwen/Qwen3.5-9B --limit 10   # 冒烟
  python probe_init_confidence.py --model Qwen/Qwen3.5-9B             # 全量 ~10min
"""
import argparse
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

import ast, base64 as b64m, io as iom
from infer_privilege import VOPD_ROOT, TSV_PATH, USER_SUFFIX, question_text, hide_compose, to_b64

ANS_DIR = f"{VOPD_ROOT}/eval/model_answer/treebench"
MAX_POS = 64
BUCKETS = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64)]


def build_prompt(item, image_b64=None):
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64 or item['image']}"},
        {"type": "text", "text": question_text(item) + USER_SUFFIX},
    ]}]


def summarize(rows):
    ok = [r for r in rows if r["correct"]]
    ng = [r for r in rows if not r["correct"]]
    print(f"\n===== 初始化置信度: 对 {len(ok)} / 错 {len(ng)}")

    def agg(rs, key, lo, hi):
        vals = [v for r in rs for v in r[key][lo:hi]]
        return sum(vals) / len(vals) if vals else float("nan")

    for key, label in [("p1", "top-1 概率"), ("margin", "margin(p1-p2)"),
                       ("entropy", "熵(nats)"), ("top5", "top-5 质量")]:
        line = f"{label:<16}"
        for lo, hi in BUCKETS:
            line += f"  [{lo:>2}-{hi:<2}) 对{agg(ok, key, lo, hi):.3f}/错{agg(ng, key, lo, hi):.3f}"
        print(line)
    # 首 token 的高置信占比 (premature commitment 的最直接读数)
    for th in (0.8, 0.9, 0.95):
        a = sum(1 for r in ok if r["p1"][0] >= th) / len(ok) * 100
        b = sum(1 for r in ng if r["p1"][0] >= th) / len(ng) * 100
        print(f"首token p1>={th}: 对 {a:.1f}%  错 {b:.1f}%")
    # 前8个位置里的"高置信锐分叉"密度: p1>=0.9 且 margin>=0.8
    a = sum(sum(1 for p, m in zip(r["p1"][:8], r["margin"][:8]) if p >= 0.9 and m >= 0.8) for r in ok) / len(ok)
    b = sum(sum(1 for p, m in zip(r["p1"][:8], r["margin"][:8]) if p >= 0.9 and m >= 0.8) for r in ng) / len(ng)
    print(f"前8位高置信位置数/链: 对 {a:.2f}  错 {b:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--traj-from", default=None)
    ap.add_argument("--privilege", choices=["none", "hide"], default="none",
                    help="hide = 用 hide 图作条件方 (轨迹默认取 hide-nosent 文件), 测特权链的熵剖面")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tag = args.model.rstrip("/").split("/")[-1].lower()
    default_traj = (f"{ANS_DIR}/{tag}-priv-hide-nosent-nothink_answer.jsonl"
                    if args.privilege == "hide" else f"{ANS_DIR}/{tag}-direct-nothink_answer.jsonl")
    traj_path = args.traj_from or default_traj
    trajs = {r["index"]: r for r in map(json.loads, open(traj_path))}
    suffix = "-hide" if args.privilege == "hide" else ""
    out_path = args.out or f"{ANS_DIR}/{tag}-init-confidence{suffix}.jsonl"
    print(f"trajectories: {traj_path} ({len(trajs)})\nout -> {out_path}")

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
    for item in tqdm(df, desc=f"{tag}-init"):
        if item["index"] in done or item["index"] not in trajs:
            continue
        traj = trajs[item["index"]]
        output = traj["output"].split("<|im_end|>")[0]
        if not output.strip():
            continue
        cont_ids = torch.tensor(tokenizer(output, add_special_tokens=False)["input_ids"])
        T = min(cont_ids.shape[0], MAX_POS)

        img_b64 = None
        if args.privilege == "hide":
            pil = Image.open(iom.BytesIO(b64m.b64decode(item["image"])))
            img_b64 = to_b64(hide_compose(pil, ast.literal_eval(item["target_instances"])))
        messages = build_prompt(item, img_b64)
        image_inputs, _ = process_vision_info(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        inputs = processor(text=[text], images=image_inputs, padding=True,
                           return_tensors="pt").to(model.device)
        full_ids = torch.cat([inputs["input_ids"], cont_ids.unsqueeze(0).to(model.device)], dim=1)
        kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        if "mm_token_type_ids" in kwargs:
            pad = torch.zeros((1, cont_ids.shape[0]), dtype=kwargs["mm_token_type_ids"].dtype,
                              device=kwargs["mm_token_type_ids"].device)
            kwargs["mm_token_type_ids"] = torch.cat([kwargs["mm_token_type_ids"], pad], dim=1)
        with torch.inference_mode():
            out = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), **kwargs)
        p0 = inputs["input_ids"].shape[1]
        logits = out.logits[0, p0 - 1: p0 - 1 + T].float()   # 预测前 T 个续写 token
        probs = torch.softmax(logits, dim=-1)
        top5 = probs.topk(5, dim=-1)
        p1 = top5.values[:, 0]
        p2 = top5.values[:, 1]
        ent = -(probs * (probs + 1e-12).log()).sum(-1)
        actual = cont_ids[:T].to(probs.device)
        rank = (probs > probs.gather(-1, actual[:, None])).sum(-1)   # 0 = 实际token即top1

        rec = {
            "index": item["index"], "category": item["category"],
            "correct": traj["prediction"].upper() == traj["answer"].upper(),
            "n_pos": int(T),
            "p1": [round(float(v), 4) for v in p1],
            "margin": [round(float(a - b), 4) for a, b in zip(p1, p2)],
            "entropy": [round(float(v), 4) for v in ent],
            "top5": [round(float(v), 4) for v in top5.values.sum(-1)],
            "actual_rank": [int(v) for v in rank],
        }
        done[item["index"]] = rec
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    summarize(list(done.values()))


if __name__ == "__main__":
    main()
