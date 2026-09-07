"""Probe B: 答案 token advantage 噪声 — hide 特权 teacher 版 OPD advantage
u = log p^H − log p^F 在答案 token 上的符号 vs 判卷正误 (论文 §2 噪声分析的视觉域复现),
外加 §3.1 近零 advantage 统计所需的逐 token dump。

学生轨迹 = gen_rollouts_treebench.py 的 T=1.0 采样 (每题取 1 正确 + 1 错误, 若有;
对齐论文 setup)。F 视图 = 原图 direct; H 视图 = hide_compose 黑底原位图 (冠军臂
hide-nosent: 无特权句, 文本与 F 逐字节相同, 图像是唯一变量)。同一模型双 teacher-forcing。

运行:
  python probe_answer_noise.py --model Qwen/Qwen3.5-4B --rollouts results/rollouts_qwen3.5-4b_t1.jsonl
  python probe_answer_noise.py --model Qwen/Qwen3.5-9B --rollouts results/rollouts_qwen3.5-9b_t1.jsonl
"""
import argparse
import ast
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
from infer_privilege import TSV_PATH, build  # noqa: E402
from _kernels import forced_stats  # noqa: E402

OPSA_DIR = os.path.dirname(os.path.abspath(__file__))


def answer_token_pos(tokenizer, token_ids):
    """答案字母 (<answer>X) 所在 token 位置。用逐 token decode 重建 char offset,
    避免 retokenize 不一致。返回 (pos, letter) 或 (None, None)。"""
    pieces = [tokenizer.decode([t], skip_special_tokens=False,
                               clean_up_tokenization_spaces=False) for t in token_ids]
    text = "".join(pieces)
    m = re.search(r"<answer>\s*([A-Z])", text)
    if not m:
        return None, None
    char_idx = m.start(1)
    off = 0
    for i, p in enumerate(pieces):
        if off <= char_idx < off + len(p):
            return i, m.group(1)
        off += len(p)
    return None, None


def pick_samples(rolls):
    """每题取 1 正确 + 1 错误 (若有); 无 prediction 的样本跳过。"""
    by_idx = {}
    for r in rolls:
        if not r["prediction"]:
            continue
        by_idx.setdefault(r["index"], []).append(r)
    picked = []
    for idx, rs in by_idx.items():
        cor = [r for r in rs if r["correct"]]
        inc = [r for r in rs if not r["correct"]]
        if cor:
            picked.append(cor[0])
        if inc:
            picked.append(inc[0])
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpus", type=int, default=1,
                    help=">1 时单命令多卡: 自动 fork N 个子进程各绑一张卡跑分片, 完成后自动合并")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    assert 0 <= args.shard < args.num_shards

    roll_path = args.rollouts if os.path.isabs(args.rollouts) else os.path.join(OPSA_DIR, args.rollouts)
    tag = args.model.rstrip("/").split("/")[-1].lower()
    shard_sfx = f".shard{args.shard}" if args.num_shards > 1 else ""
    out_path = args.out or os.path.join(OPSA_DIR, "results", f"answer_noise_rows_{tag}{shard_sfx}.jsonl")
    if not os.path.isabs(out_path):
        out_path = os.path.join(OPSA_DIR, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    from _multi_gpu import fanout_and_merge
    if fanout_and_merge(args.gpus, out_path):
        return

    df = load_dataset("csv", data_files=TSV_PATH, delimiter="\t")["train"]
    items = {it["index"]: it for it in df}

    rolls = [json.loads(l) for l in open(roll_path)]
    picked = pick_samples(rolls)
    if args.limit:
        picked = picked[: args.limit]
    if args.num_shards > 1:
        picked = picked[args.shard :: args.num_shards]
    n_cor = sum(1 for r in picked if r["correct"])
    print(f"picked {len(picked)} chains ({n_cor} correct / {len(picked)-n_cor} wrong) "
          f"from {len(rolls)} rollouts")

    done = set()
    if os.path.exists(out_path):
        done = {(r["index"], r["sample_id"]) for r in map(json.loads, open(out_path))}
        print(f"resume: {len(done)}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer

    n_ocr = n_noans = 0
    out_f = open(out_path, "a")
    for r in tqdm(picked, desc=f"answer-noise-{tag}"):
        key = (r["index"], r["sample_id"])
        if key in done or r["n_tokens"] < 5:
            continue
        item = items[r["index"]]
        if item["category"] == "OCR":
            n_ocr += 1
            continue
        pil_img = Image.open(io.BytesIO(base64.b64decode(item["image"])))
        w, h = pil_img.size
        cont_ids = torch.tensor(r["token_ids"])
        ans_pos, ans_letter = answer_token_pos(tokenizer, r["token_ids"])
        if ans_pos is None:
            n_noans += 1

        def encode(privilege, no_sentence):
            messages, _ = build(item, privilege, w, h, None, pil_img=pil_img,
                                no_sentence=no_sentence)
            image_inputs, video_inputs = process_vision_info(messages)
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True,
                                                 enable_thinking=False)
            return processor(text=[text], images=image_inputs, videos=video_inputs,
                             padding=True, return_tensors="pt").to(model.device)

        try:
            in_F = encode("none", False)
            in_H = encode("hide", True)   # hide-nosent 冠军臂: 文本与 F 相同
            lr_F, ent_F, _, _ = forced_stats(model, in_F, cont_ids, want_ent=True, want_topk=False)
            lr_H, _, _, _ = forced_stats(model, in_H, cont_ids, want_ent=False, want_topk=False)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"[{key}] 失败, 跳过 (重跑同命令会重试)", flush=True)
            torch.cuda.empty_cache()
            continue

        rec = {
            "index": r["index"], "sample_id": r["sample_id"], "category": r["category"],
            "correct": r["correct"], "answer": r["answer"], "prediction": r["prediction"],
            "answer_pos": ans_pos, "answer_letter": ans_letter,
            "n_tokens": r["n_tokens"], "model": args.model,
            "tok": r["token_ids"],
            "lrF": [round(float(v), 4) for v in lr_F],
            "lrH": [round(float(v), 4) for v in lr_H],
            "entF": [round(float(v), 4) for v in ent_F],
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    print(f"saved -> {out_path}  (skipped OCR: {n_ocr}, 无答案token: {n_noans})")


if __name__ == "__main__":
    main()
