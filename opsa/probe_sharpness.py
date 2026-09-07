"""Probe A: 分布尖锐度 — 对 T=1.0 采样轨迹 teacher-force, 逐 token 持久化
logp / entropy(full-vocab) / top-5 候选 (id+prob).

输入 = gen_rollouts_treebench.py 的输出 (token_ids 精确保存, 零 retokenize 误差)。
统计由 summarize.py 在 CPU 上出 (四组: 熵分布 / tail-20% 熵 / 高熵位反思词 / tail 构成)。

运行:
  python probe_sharpness.py --model Qwen/Qwen3.5-4B --rollouts results/rollouts_qwen3.5-4b_t1.jsonl
  python probe_sharpness.py --model Qwen/Qwen3.5-9B --rollouts results/rollouts_qwen3.5-9b_t1.jsonl
"""
import argparse
import base64
import io
import json
import os
import sys

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "eval", "treebench_probe"))
from infer_privilege import TSV_PATH, build  # noqa: E402

OPSA_DIR = os.path.dirname(os.path.abspath(__file__))


from _kernels import forced_stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rollouts", required=True, help="gen_rollouts_treebench.py 输出 jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条 (冒烟)")
    ap.add_argument("--gpus", type=int, default=1,
                    help=">1 时单命令多卡: 自动 fork N 个子进程各绑一张卡跑分片, 完成后自动合并")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    assert 0 <= args.shard < args.num_shards

    roll_path = args.rollouts if os.path.isabs(args.rollouts) else os.path.join(OPSA_DIR, args.rollouts)
    tag = args.model.rstrip("/").split("/")[-1].lower()
    shard_sfx = f".shard{args.shard}" if args.num_shards > 1 else ""
    out_path = args.out or os.path.join(OPSA_DIR, "results", f"sharpness_rows_{tag}{shard_sfx}.jsonl")
    if not os.path.isabs(out_path):
        out_path = os.path.join(OPSA_DIR, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    from _multi_gpu import fanout_and_merge
    if fanout_and_merge(args.gpus, out_path):
        return

    df = load_dataset("csv", data_files=TSV_PATH, delimiter="\t")["train"]
    items = {it["index"]: it for it in df}

    rolls = [json.loads(l) for l in open(roll_path)]
    if args.limit:
        rolls = rolls[: args.limit]
    if args.num_shards > 1:
        rolls = rolls[args.shard :: args.num_shards]
    print(f"rollouts: {len(rolls)} from {roll_path}"
          + (f" (shard {args.shard}/{args.num_shards})" if args.num_shards > 1 else ""))

    done = set()
    if os.path.exists(out_path):
        done = {(r["index"], r["sample_id"]) for r in map(json.loads, open(out_path))}
        print(f"resume: {len(done)}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)

    out_f = open(out_path, "a")
    failed = []
    for r in tqdm(rolls, desc=f"sharpness-{tag}"):
        key = (r["index"], r["sample_id"])
        if key in done or r["n_tokens"] < 5:
            continue
        try:
            item = items[r["index"]]
            pil_img = Image.open(io.BytesIO(base64.b64decode(item["image"])))
            w, h = pil_img.size
            messages, _ = build(item, "none", w, h, None, pil_img=pil_img)
            image_inputs, video_inputs = process_vision_info(messages)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to(model.device)
            cont_ids = torch.tensor(r["token_ids"])
            lr, ent, t5i, t5p = forced_stats(model, inputs, cont_ids)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"[{key}] 失败, 跳过 (重跑同命令会重试)", flush=True)
            failed.append(key)
            torch.cuda.empty_cache()
            continue
        rec = {
            "index": r["index"], "sample_id": r["sample_id"], "category": r["category"],
            "correct": r["correct"], "n_tokens": r["n_tokens"], "model": args.model,
            "tok": r["token_ids"],
            "logp": [round(float(v), 4) for v in lr],
            "ent": [round(float(v), 4) for v in ent],
            "top5_ids": t5i.tolist(),
            "top5_p": [[round(float(v), 4) for v in row] for row in t5p],
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    if failed:
        print(f"⚠️ 失败 {len(failed)} 条 (已跳过): {failed}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
