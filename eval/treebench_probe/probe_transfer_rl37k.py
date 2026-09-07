"""RL-37K 过户验证: TreeBench 上的形态排序与残差方向是否迁移到训练分布。

评分 = GT 答案的长度归一化 logp (每条件一次 forward, 免生成免文本匹配 ——
且比 acc 更贴 OPD 语义: 训练优化的就是分布)。

条件 (nothink, 文本 prompt 各条件逐字节相同):
  direct  原图 (基线)
  hide    黑底原位融合, 纯图无句 (9B 冠军形态)
  draw    全图+红框+标签句 (4B 冠军形态; 标签用 RL-37K 原生 name)
  region  2.42x 裁剪+红框 (训练历史形态)

读数:
  各条件 mean Δlogp(GT answer) vs direct + win rate (条件>direct 的题占比)
  —— 形态排序应复现 TreeBench 结论 (hide/draw > region > direct);
  hide−direct 即残差方向的训练分布版, 应为正。
坐标空间自动判定: RL-37K 标称绝对像素, 用越界率校验。

运行 (vision-opd 环境, GPU; 600 题 x 4 条件 ~35min):
  python probe_transfer_rl37k.py --model Qwen/Qwen3.5-9B --n 600
"""
import argparse
import json
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from infer_privilege import (VOPD_ROOT, hide_compose, draw_gt, region_crop, to_b64,
                             DRAW_SENTENCE_LABELED, REGION_SENTENCE)

DATA_DIR = f"{VOPD_ROOT}/data/TreeVGR-RL-37K"
ANS_DIR = f"{VOPD_ROOT}/eval/model_answer/treebench"
CONDS = ["direct", "hide", "draw", "region"]
MAX_ANS_TOKENS = 64


def get_boxes_names(row):
    boxes, names = [], []
    for inst in row["target_instances"]:
        b = [float(v) for v in inst["bbox"]]
        boxes.append(b)
        names.append(str(inst.get("name") or ""))
    return boxes, names


def build_images(pil, boxes, names):
    out = {"direct": [None], "hide": None, "draw": None, "region": None}
    out["hide"] = [to_b64(hide_compose(pil, boxes))]
    out["draw"] = [to_b64(draw_gt(pil, boxes))]
    out["region"] = [to_b64(region_crop(pil, b)) for b in boxes]
    return out


def cond_text(cond, question, names):
    # v2: 全条件文本逐字节相同 (纯问题), 特权只经图像通道 —— 消除标签句点名答案名词的混杂
    return question


def summarize(rows):
    import statistics
    print(f"\n===== RL-37K 过户验证: {len(rows)} 题")
    base = {r["orig_row"]: r["logp"]["direct"] for r in rows}
    for c in CONDS:
        d = [r["logp"][c] - base[r["orig_row"]] for r in rows]
        win = sum(1 for v in d if v > 0) / len(d) * 100
        print(f"  {c:<8} Δlogp(GT) vs direct: {statistics.mean(d):+.4f}   win {win:.1f}%")
    bytask = {}
    for r in rows:
        if r.get("task"):
            bytask.setdefault(r["task"], []).append(r)
    for tname, sub2 in sorted(bytask.items(), key=lambda x: -len(x[1])):
        if len(sub2) < 15:
            continue
        line = f"  [task={tname} n={len(sub2)}]"
        for c in ("hide", "draw", "region"):
            d = [r["logp"][c] - r["logp"]["direct"] for r in sub2]
            line += f"  {c} {statistics.mean(d):+.4f}"
        print(line)
    # 分段 (前30k=V* / 后段=VisDrone)
    for name, pred in [("V*段", lambda r: r["orig_row"] < 30000), ("VisDrone段", lambda r: r["orig_row"] >= 30000)]:
        sub = [r for r in rows if pred(r)]
        if len(sub) < 20:
            continue
        line = f"  [{name} n={len(sub)}]"
        for c in ("hide", "draw", "region"):
            d = [r["logp"][c] - r["logp"]["direct"] for r in sub]
            line += f"  {c} {statistics.mean(d):+.4f}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--subset", default="subset6k.parquet", help="如 subset4k_adv.parquet")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = args.model.rstrip("/").split("/")[-1].lower()
    suffix = "v2" if args.subset.startswith("subset6k") else "adv"
    out_path = args.out or f"{ANS_DIR}/{tag}-rl37k-transfer-{suffix}.jsonl"
    sub = pd.read_parquet(f"{DATA_DIR}/{args.subset}")
    sub = sub.sample(frac=1.0, random_state=43).reset_index(drop=True)
    probe = sub.iloc[:args.n]
    print(f"probe rows: {len(probe)}; out -> {out_path}")

    done = {}
    if os.path.exists(out_path):
        done = {r["orig_row"]: r for r in map(json.loads, open(out_path))}
        print(f"resume: {len(done)}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer

    n_over = n_box = 0
    out_f = open(out_path, "a")
    for _, row in tqdm(probe.iterrows(), total=len(probe), desc=f"{tag}-transfer"):
        orig = int(row["orig_row"])
        if orig in done:
            continue
        img_path = f"{DATA_DIR}/{row['images'][0]}"
        if not os.path.exists(img_path):
            continue
        pil = Image.open(img_path).convert("RGB")
        w, h = pil.size
        boxes, names = get_boxes_names(row)
        n_box += len(boxes)
        n_over += sum(1 for b in boxes if max(b) > max(w, h) * 1.02)
        question = str(row["problem"]).replace("<image>", "").strip()
        answer = str(row["answer"]).strip()
        ans_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"][:MAX_ANS_TOKENS]
        ans_t = torch.tensor(ans_ids)

        images = build_images(pil, boxes, names)
        logps = {}
        for c in CONDS:
            content = []
            for ib in (images[c] if images[c] != [None] else [None]):
                if ib is None:
                    import base64 as b64m, io as iom
                    buf = iom.BytesIO(); pil.save(buf, format="JPEG", quality=92)
                    ib = b64m.b64encode(buf.getvalue()).decode("ascii")
                content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{ib}"})
            content.append({"type": "text", "text": cond_text(c, question, names)})
            messages = [{"role": "user", "content": content}]
            image_inputs, _ = process_vision_info(messages)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            inputs = processor(text=[text], images=image_inputs, padding=True,
                               return_tensors="pt").to(model.device)
            full_ids = torch.cat([inputs["input_ids"], ans_t.unsqueeze(0).to(model.device)], dim=1)
            kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
            if "mm_token_type_ids" in kwargs:
                pad = torch.zeros((1, ans_t.shape[0]), dtype=kwargs["mm_token_type_ids"].dtype,
                                  device=kwargs["mm_token_type_ids"].device)
                kwargs["mm_token_type_ids"] = torch.cat([kwargs["mm_token_type_ids"], pad], dim=1)
            with torch.inference_mode():
                out = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), **kwargs)
            p0 = inputs["input_ids"].shape[1]
            lp = torch.log_softmax(out.logits[0, p0 - 1: p0 - 1 + ans_t.shape[0]].float(), dim=-1)
            tok_lp = lp.gather(-1, ans_t.to(lp.device).unsqueeze(-1)).squeeze(-1)
            logps[c] = round(float(tok_lp.mean()), 4)

        rec = {"orig_row": orig, "n_boxes": len(boxes), "logp": logps,
               "task": str(row["task"]) if "task" in row else None}
        done[orig] = rec
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    print(f"坐标越界率: {n_over}/{n_box} (>2% 则非绝对像素, 需换算)")
    summarize(list(done.values()))


if __name__ == "__main__":
    main()
