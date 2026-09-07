"""Vision-OPD-6K 过滤池的 hide 头寸探针.

在 frac>=阈值 且高清的 6K 行上抽样, 对比同一 4B 模型三个视角的可答率:
  full = 干净原图 (学生视角基线; original_images, 无红框)
  hide = hide_compose(原图, [bbox]) 冠军渲染 (候选教师视角)
头寸 = hide - full. RL-37K 上头寸≈0 判死了那条路; 这里决定 6K+hide 训练格子的生死.

问题用 extra_info.question (干净版, 无 focus 句). 标签仅用于离线测量 (允许).

用法:
  python probe_6k_hide.py --model <4B路径> --sample 300            # 两视角都跑
  python probe_6k_hide.py --model <4B路径> --sample 300 --views hide
"""
import argparse
import json
import os
import re


def extract_letter(out, question):
    """多模式抽取; 全失败时用选项文本反查 (末段唯一命中才算)."""
    out = out.strip()
    for p in [r"<answer>\s*\(?([A-D])", r"[Aa]nswer(?:\s+is)?[:\s]*\**\(?([A-D])\)?[.\s)]",
              r"\*\*\(?([A-D])\)?(?:[.):\s]|\*)", r"^\(?([A-D])\)?(?:[.):\s]|$)"]:
        m = re.search(p, out)
        if m:
            return m.group(1).upper()
    ms = list(re.finditer(r"\b([A-D])[.)]", out))
    if ms:
        return ms[-1].group(1).upper()
    opts = dict(re.findall(r"^([A-D])\.\s*(.+)$", question, re.M))
    tail = out[-250:].lower()
    hits = [L for L, txt in opts.items() if txt.strip().lower().rstrip(".") in tail]
    return hits[0] if len(hits) == 1 else ""

from PIL import Image
from tqdm import tqdm

from infer_privilege import hide_compose, to_b64

Image.MAX_IMAGE_PIXELS = None
VOPD_ROOT = os.environ.get("VOPD_ROOT", "/scratch/nkw3mr/Vision-OPD")


def load_pool(min_frac, min_vtok):
    rows = [json.loads(l) for l in open(f"{VOPD_ROOT}/data/train.jsonl")]
    pool = []
    for i, r in enumerate(rows):
        b = r["bbox"]
        p = f"{VOPD_ROOT}/data/{r['original_images'][0]}"
        W, H = Image.open(p).size
        frac = max(0, b[2] - b[0]) * max(0, b[3] - b[1]) / (W * H) * 100
        vtok = W * H // (28 * 28 * 4)
        if frac >= min_frac and vtok >= min_vtok:
            pool.append({"row": i, "img": p, "bbox": b,
                         "question": r["extra_info"]["question"],
                         "answer": r["extra_info"]["answer"], "frac": frac})
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-frac", type=float, default=1.0)
    ap.add_argument("--min-vtok", type=int, default=1000)
    ap.add_argument("--views", default="full,hide")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    pool = load_pool(args.min_frac, args.min_vtok)
    print(f"过滤池: {len(pool)} 行 (frac>={args.min_frac}%, vtok>={args.min_vtok})")
    import random
    random.Random(args.seed).shuffle(pool)
    pool = pool[:args.sample]

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)

    out_dir = f"{VOPD_ROOT}/eval/treebench_probe"
    for view in args.views.split(","):
        out_path = f"{out_dir}/probe6k_{view}_s{args.seed}.jsonl"
        done = set()
        if os.path.exists(out_path):
            done = {json.loads(l)["row"] for l in open(out_path)}
        out_f = open(out_path, "a")
        n_ok = n = 0
        for it in tqdm(pool, desc=view):
            if it["row"] in done:
                continue
            img = Image.open(it["img"]).convert("RGB")
            shown = hide_compose(img, [it["bbox"]]) if view == "hide" else img
            content = [{"type": "image_url", "image_url": f"data:image/jpeg;base64,{to_b64(shown)}"},
                       {"type": "text", "text": it["question"]}]
            messages = [{"role": "user", "content": content}]
            image_inputs, video_inputs = process_vision_info(messages)
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                gen = model.generate(**inputs, top_p=0.001, top_k=1, temperature=0.01,
                                     repetition_penalty=1.0, max_new_tokens=args.max_new_tokens,
                                     use_cache=True, do_sample=True)
            out = processor.batch_decode([gen[0][inputs.input_ids.shape[1]:]],
                                         skip_special_tokens=True)[0].strip()
            pred = extract_letter(out, it["question"])
            ok = pred == it["answer"].strip().upper()
            n += 1; n_ok += ok
            out_f.write(json.dumps({"row": it["row"], "view": view, "frac": it["frac"],
                                    "answer": it["answer"], "pred": pred, "ok": ok,
                                    "output": out}, ensure_ascii=False) + "\n")
            out_f.flush()
        out_f.close()
        allr = [json.loads(l) for l in open(out_path)]
        acc = sum(r["ok"] for r in allr) / len(allr) * 100
        n_empty = sum(1 for r in allr if not r["pred"])
        print(f"== {view}: {sum(r['ok'] for r in allr)}/{len(allr)} = {acc:.2f}%  (抽取失败 {n_empty})  -> {out_path}")


if __name__ == "__main__":
    main()
