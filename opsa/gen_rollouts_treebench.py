"""Phase 0: TreeBench 405 题 T=1.0 on-policy 采样轨迹生成 (OPSA probe 共用前置).

与 eval/treebench_probe/infer_privilege.py 的 direct(nothink) 臂同 prompt 构造
(privilege=none, enable_thinking=False), 唯一差别是解码参数改为采样并每题采 n 条。
直接持久化生成的 token id 序列 (截到首个 <|im_end|> 含), 后续 probe teacher-forcing
时零 retokenize 误差。

运行:
  python gen_rollouts_treebench.py --model Qwen/Qwen3.5-4B
  python gen_rollouts_treebench.py --model Qwen/Qwen3.5-9B
冒烟 (应与已有 greedy 轨迹 prediction 一致):
  python gen_rollouts_treebench.py --model Qwen/Qwen3.5-4B --limit 3 --n 1 \
      --temperature 0.01 --top-p 0.001 --top-k 1 --out results/smoke_4b.jsonl
"""
import argparse
import base64
import io
import json
import os
import re
import sys

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "eval", "treebench_probe"))
from infer_privilege import VOPD_ROOT, TSV_PATH, build  # noqa: E402

OPSA_DIR = os.path.dirname(os.path.abspath(__file__))


def trim_at_eos(ids, eos_ids):
    """截到首个 eos (含); 去掉 batch 生成的尾部 pad."""
    for i, t in enumerate(ids):
        if t in eos_ids:
            return ids[: i + 1]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=4, help="每题采样条数")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0, help="0 或 -1 = 不限")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpus", type=int, default=1,
                    help=">1 时单命令多卡: 自动 fork N 个子进程各绑一张卡跑分片, 完成后自动合并")
    ap.add_argument("--shard", type=int, default=0, help="本进程分片号 (0-based; --gpus 会自动传)")
    ap.add_argument("--num-shards", type=int, default=1, help="手动分片总数 (--gpus 会自动传)")
    args = ap.parse_args()
    top_k = 0 if args.top_k < 0 else args.top_k
    assert 0 <= args.shard < args.num_shards

    tag = args.model.rstrip("/").split("/")[-1].lower()
    shard_sfx = f".shard{args.shard}" if args.num_shards > 1 else ""
    out_path = args.out or os.path.join(OPSA_DIR, "results", f"rollouts_{tag}_t1{shard_sfx}.jsonl")
    if not os.path.isabs(out_path):
        out_path = os.path.join(OPSA_DIR, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    from _multi_gpu import fanout_and_merge
    if fanout_and_merge(args.gpus, out_path):
        summarize_out(out_path)
        return

    df = load_dataset("csv", data_files=TSV_PATH, delimiter="\t")["train"]
    if args.limit:
        df = df.select(range(args.limit))
    if args.num_shards > 1:
        df = df.select(range(args.shard, len(df), args.num_shards))
        print(f"shard {args.shard}/{args.num_shards}: {len(df)} items")

    done = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done.setdefault(r["index"], set()).add(r["sample_id"])
        print(f"resume: {sum(len(v) for v in done.values())} samples over {len(done)} items")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    eos_ids = set(tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")
                  if tokenizer.convert_tokens_to_ids(t) is not None)

    out_f = open(out_path, "a")
    failed = []
    for item in tqdm(df, desc=f"gen-{tag}"):
        have = done.get(item["index"], set())
        if len(have) >= args.n:
            continue
        try:
            pil_img = Image.open(io.BytesIO(base64.b64decode(item["image"])))
            w, h = pil_img.size
            messages, prefill = build(item, "none", w, h, None, pil_img=pil_img)
            image_inputs, video_inputs = process_vision_info(messages)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False) + prefill
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to(model.device)
            # 每题固定种子, 与 resume 顺序无关, 可复现
            torch.manual_seed(args.seed * 1_000_000 + int(item["index"]))
            gen_kw = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p,
                          top_k=top_k, repetition_penalty=1.0,
                          max_new_tokens=args.max_new_tokens, use_cache=True)
            try:
                with torch.inference_mode():
                    gen = model.generate(**inputs, num_return_sequences=args.n, **gen_kw)
            except torch.cuda.OutOfMemoryError:
                # 大图 OOM 降级: 逐条生成
                torch.cuda.empty_cache()
                print(f"\n[index {item['index']}] n={args.n} 批量生成 OOM, 降级逐条重试", flush=True)
                seqs = []
                for sid in range(args.n):
                    torch.manual_seed(args.seed * 1_000_000 + int(item["index"]) + 7919 * (sid + 1))
                    with torch.inference_mode():
                        g1 = model.generate(**inputs, num_return_sequences=1, **gen_kw)
                    seqs.append(g1[0])
                maxlen = max(s.shape[0] for s in seqs)
                gen = torch.full((args.n, maxlen), list(eos_ids)[0], dtype=seqs[0].dtype,
                                 device=seqs[0].device)
                for sid, s in enumerate(seqs):
                    gen[sid, : s.shape[0]] = s
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"[index {item['index']}] 失败, 跳过 (重跑同命令会重试)", flush=True)
            failed.append(item["index"])
            torch.cuda.empty_cache()
            continue
        p0 = inputs.input_ids.shape[1]
        for sid in range(args.n):
            if sid in have:
                continue
            ids = trim_at_eos(gen[sid][p0:].tolist(), eos_ids)
            output_text = tokenizer.decode(ids, skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False)
            m = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
            pred = m.group(1).strip().upper() if m else ""
            rec = {
                "index": item["index"], "sample_id": sid, "category": item["category"],
                "answer": item["answer"], "prediction": pred,
                "correct": bool(pred and item["answer"] and
                                pred[:1] == str(item["answer"]).strip().upper()[:1]),
                "image_size": [w, h], "model": args.model,
                "temperature": args.temperature, "n_tokens": len(ids),
                "token_ids": ids, "output": output_text,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()

    if failed:
        print(f"⚠️ 失败 {len(failed)} 题 (已跳过): {failed}; 重跑同命令自动重试", flush=True)
    summarize_out(out_path)


def summarize_out(out_path):
    rows = [json.loads(l) for l in open(out_path)]
    n_ok = sum(1 for r in rows if r["correct"])
    n_fmt = sum(1 for r in rows if not r["prediction"])
    print(f"==> {len(rows)} samples, acc={n_ok}/{len(rows)}={n_ok/max(len(rows),1)*100:.2f}%, "
          f"无<answer>: {n_fmt}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
