"""Probe A 校准锚: Qwen3-1.7B + AIME24 (论文已证明 OPSA 有效的域), T=1.0 采样 +
同一套逐 token logp/entropy/top-5 dump, 输出 schema 与 probe_sharpness.py 一致
(供 summarize.py 同口径对比视觉域的分布尖锐度)。

数据: zhuzilin/aime-2024 (30 题, 字段 prompt/label, 官方 OPSA README 同款)。
模型与数据首跑自动从 HF Hub 下载 (计算节点可联网)。

运行:
  python probe_aime_anchor.py                    # 默认 Qwen/Qwen3-1.7B, n=4
  python probe_aime_anchor.py --limit 2 --n 1    # 冒烟
"""
import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

OPSA_DIR = os.path.dirname(os.path.abspath(__file__))


def forced_stats_text(model, prompt_ids, cont_ids, topk=5, chunk=128):
    full_ids = torch.cat([prompt_ids, cont_ids.unsqueeze(0).to(model.device)], dim=1)
    with torch.inference_mode():
        out = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids))
    p0 = prompt_ids.shape[1]
    logits = out.logits[0, p0 - 1: p0 - 1 + cont_ids.shape[0]]
    T = cont_ids.shape[0]
    lr, ent = torch.empty(T), torch.empty(T)
    t5i = torch.empty((T, topk), dtype=torch.long)
    t5p = torch.empty((T, topk))
    for i in range(0, T, chunk):
        lp = torch.log_softmax(logits[i:i + chunk].float(), dim=-1)
        lr[i:i + chunk] = lp.gather(-1, cont_ids[i:i + chunk, None].to(lp.device)).squeeze(-1).cpu()
        ent[i:i + chunk] = (-(lp.exp() * lp).sum(-1)).cpu()
        tv, ti = lp.exp().topk(topk, dim=-1)
        t5i[i:i + chunk] = ti.cpu()
        t5p[i:i + chunk] = tv.cpu()
    return lr, ent, t5i, t5p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    top_k = 0 if args.top_k < 0 else args.top_k

    from huggingface_hub import hf_hub_download
    data_path = hf_hub_download(repo_id="zhuzilin/aime-2024", filename="aime-2024.jsonl",
                                repo_type="dataset")
    problems = [json.loads(l) for l in open(data_path)]
    if args.limit:
        problems = problems[: args.limit]
    print(f"AIME24: {len(problems)} problems from {data_path}")

    tag = args.model.rstrip("/").split("/")[-1].lower()
    out_path = args.out or os.path.join(OPSA_DIR, "results", f"sharpness_rows_{tag}_aime.jsonl")
    if not os.path.isabs(out_path):
        out_path = os.path.join(OPSA_DIR, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    done = set()
    if os.path.exists(out_path):
        done = {(r["index"], r["sample_id"]) for r in map(json.loads, open(out_path))}
        print(f"resume: {len(done)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eos_ids = {tokenizer.eos_token_id}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None:
        eos_ids.add(im_end)

    out_f = open(out_path, "a")
    for qi, prob in enumerate(tqdm(problems, desc=f"aime-{tag}")):
        if all((qi, sid) in done for sid in range(args.n)):
            continue
        prompt = prob["prompt"] if isinstance(prob["prompt"], str) else str(prob["prompt"])
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        prompt_ids = tokenizer([text], return_tensors="pt").input_ids.to(model.device)
        torch.manual_seed(args.seed * 1_000_000 + qi)
        with torch.inference_mode():
            gen = model.generate(input_ids=prompt_ids, do_sample=True,
                                 temperature=args.temperature, top_p=args.top_p, top_k=top_k,
                                 repetition_penalty=1.0, num_return_sequences=args.n,
                                 max_new_tokens=args.max_new_tokens, use_cache=True)
        p0 = prompt_ids.shape[1]
        for sid in range(args.n):
            if (qi, sid) in done:
                continue
            ids = gen[sid][p0:].tolist()
            for i, t in enumerate(ids):
                if t in eos_ids:
                    ids = ids[: i + 1]
                    break
            if len(ids) < 5:
                continue
            cont_ids = torch.tensor(ids)
            lr, ent, t5i, t5p = forced_stats_text(model, prompt_ids, cont_ids)
            rec = {
                "index": qi, "sample_id": sid, "category": "aime24",
                "label": str(prob.get("label", "")), "correct": None,
                "n_tokens": len(ids), "model": args.model,
                "tok": ids,
                "logp": [round(float(v), 4) for v in lr],
                "ent": [round(float(v), 4) for v in ent],
                "top5_ids": t5i.tolist(),
                "top5_p": [[round(float(v), 4) for v in row] for row in t5p],
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
    out_f.close()
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
