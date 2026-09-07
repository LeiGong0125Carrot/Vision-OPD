"""教师可答性测量 + teacher-verified 训练子集构建。

对 train_sa4k 的每一题, 让冻结教师在 H 视图(region crops, 与训练完全同构)下作答:
  贪心 1 次 -> 关键词代理判对错 (真标签版过滤依据)
  可选 --k 3 采样 -> 自一致性 (免标签版过滤依据) 及两版一致率
输出:
  eval/treebench_probe/teacher_acc_rows.jsonl  逐题记录
  汇总: 分任务教师正确率 / 自一致率 / 两版过滤一致率
  --build 时: data/TreeVGR-RL-37K/train_sa4k_tv.parquet (教师答对子集)

运行 (单卡):
  python scripts/measure_teacher_acc.py                 # 贪心测量 ~60min
  python scripts/measure_teacher_acc.py --k 3           # +自一致性 (3x 时间)
  python scripts/measure_teacher_acc.py --build         # 测量完后构建过滤集
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

VOPD = "/sfs/weka/scratch/nkw3mr/Vision-OPD"
PARQUET = f"{VOPD}/data/TreeVGR-RL-37K/train_sa4k.parquet"
OUT_TPL = VOPD + "/eval/treebench_probe/teacher_acc_rows{sfx}.jsonl"
TV_OUT = f"{VOPD}/data/TreeVGR-RL-37K/train_sa4k_tv.parquet"
STOP = set("the a an of in on at to for with and or is are was were it its this that side image there".split())


def words(s):
    return set(w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2)


def kw_correct(question, gts, output):
    ans_new = words(gts) - words(question)
    if not ans_new:
        return None
    return len(ans_new & words(output)) / len(ans_new) >= 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--k", type=int, default=0, help=">0 时额外做 k 次采样自一致性")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--view", choices=["region", "full", "hide"], default="region")
    ap.add_argument("--sample", type=int, default=0, help="随机抽 N 题 (seed 7)")
    ap.add_argument("--build", action="store_true", help="用已有测量结果构建 train_sa4k_tv.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    OUT = OUT_TPL.format(sfx="" if args.view == "region" else "_" + args.view)
    if args.sample:
        df = df.sample(n=args.sample, random_state=7).reset_index(drop=True)
    if args.view == "hide":
        import sys as _sys
        _sys.path.insert(0, f"{VOPD}/eval/treebench_probe")
        from infer_privilege import hide_compose
        adv = pd.read_parquet(f"{VOPD}/data/TreeVGR-RL-37K/subset4k_adv_v2.parquet")
        boxmap = {int(r["orig_row"]): [[float(x) for x in inst["bbox"]] for inst in r["target_instances"]]
                  for _, r in adv.iterrows()}

    if args.build:
        rows = {r["orig_row"]: r for r in map(json.loads, open(OUT))}
        keep = [i for i, row in df.iterrows()
                if rows.get(int(row["extra_info"]["orig_row"]), {}).get("greedy_correct") is True]
        sub = df.iloc[keep].reset_index(drop=True)
        sub.to_parquet(TV_OUT)
        print(f"teacher-verified 子集: {len(sub)}/{len(df)} -> {TV_OUT}")
        print(sub["extra_info"].apply(lambda e: e["task"]).value_counts())
        return

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)

    done = {}
    if os.path.exists(OUT):
        done = {r["orig_row"]: r for r in map(json.loads, open(OUT))}
        print(f"resume: {len(done)}")
    out_f = open(OUT, "a")
    it = df.iterrows() if not args.limit else list(df.iterrows())[:args.limit]
    for _, row in tqdm(it, total=(args.limit or len(df)), desc="teacher-acc"):
        orig = int(row["extra_info"]["orig_row"])
        if orig in done:
            continue
        question = str(row["extra_info"]["question"]).strip()
        gts = str(row["extra_info"]["answer"]).strip()
        if args.view == "region":
            img_content = [{"type": "image", "image": b["path"]} for b in row["bbox_images"]]
        elif args.view == "full":
            img_content = [{"type": "image", "image": row["images"][0]["path"]}]
        else:  # hide: 全图黑底 + 原位贴 2.42x 外扩区域
            from PIL import Image as PILImage
            pil = PILImage.open(row["images"][0]["path"]).convert("RGB")
            hp = f"/tmp/hide_{orig}.jpg"
            hide_compose(pil, boxmap[orig]).save(hp, quality=92)
            img_content = [{"type": "image", "image": hp}]
        messages = [{"role": "user", "content": [
            *img_content,
            {"type": "text", "text": question},
        ]}]
        image_inputs, _ = process_vision_info(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        inputs = processor(text=[text], images=image_inputs, padding=True,
                           return_tensors="pt").to(model.device)

        def gen(sample):
            with torch.inference_mode():
                if sample:
                    g = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.9,
                                       max_new_tokens=128)
                else:
                    g = model.generate(**inputs, do_sample=True, top_k=1, temperature=0.01,
                                       top_p=0.001, max_new_tokens=128)
            return processor.tokenizer.decode(g[0][inputs["input_ids"].shape[1]:],
                                              skip_special_tokens=True)

        greedy = gen(False)
        rec = dict(orig_row=orig, task=str(row["extra_info"]["task"]),
                   greedy_out=greedy[:600], greedy_correct=kw_correct(question, gts, greedy))
        if args.k > 0:
            outs = [gen(True) for _ in range(args.k)]
            rec["k_correct"] = [kw_correct(question, gts, o) for o in outs]
            ws = [words(o) for o in outs]
            agree = 0
            pairs = 0
            for a in range(len(ws)):
                for b in range(a + 1, len(ws)):
                    pairs += 1
                    inter = len(ws[a] & ws[b]); union = max(len(ws[a] | ws[b]), 1)
                    agree += (inter / union) >= 0.5
            rec["selfcons"] = agree / max(pairs, 1) >= 0.5
        done[orig] = rec
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()

    import statistics as st
    from collections import defaultdict
    rows = list(done.values())
    judged = [r for r in rows if r["greedy_correct"] is not None]
    print(f"\n===== 教师可答性 (view={args.view}): 可判 {len(judged)}/{len(rows)}")
    print(f"  总正确率(贪心, 关键词代理): {sum(r['greedy_correct'] for r in judged)/len(judged)*100:.1f}%")
    byt = defaultdict(list)
    for r in judged:
        byt[r["task"]].append(r["greedy_correct"])
    for t, v in sorted(byt.items(), key=lambda x: -len(x[1])):
        print(f"    {t:<14} {sum(v)/len(v)*100:5.1f}%  (n={len(v)})")
    if args.k > 0:
        sc = [r for r in judged if "selfcons" in r]
        print(f"  自一致率: {sum(r['selfcons'] for r in sc)/len(sc)*100:.1f}%")
        both = sum(1 for r in sc if r["selfcons"] == r["greedy_correct"])
        print(f"  免标签(自一致) vs 真标签(贪心对) 过滤一致率: {both/len(sc)*100:.1f}%")


if __name__ == "__main__":
    main()
