"""logprob 特权探针: 免训练测 OPD 信号强度 —— teacher(同模型+特权)在 student 自己的
干净链上做 forward, 答案位置的分布是否朝正确方向倾斜。

框架(与用户讨论定稿): 特权的价值不是"teacher 一次做对", 而是分布在 student 链上的
倾斜; 正确率只是倾斜越过 argmax 阈值后的量化影子。本探针直接量倾斜本身:

  链条固定(direct 轨迹逐 token 不变), prompt 三条件:
    clean    与 direct 完全一致(对照)
    priv     labeled box_block(与 infer_privilege.py 逐字符同源, 8B 上 acc +3.70 的形态)
    placebo  同格式句子 + 真标签 + 随机假坐标(误导对照: 分离"多了一句话"vs"真实信息")

  主读数(答案字母位置, 仅多选题):
    margin = logp(正确字母) - max logp(其他选项字母)
    Δmargin(priv-clean) 按 direct 对/错子集分开 —— 错题子集均值 = 可蒸救回潜力,
    对题子集均值 = 破坏风险; placebo 两者都应 ≈0 才能读因果。
    flip = 三条件下选项字母受限 argmax 的翻转(错->对 / 对->错)。
  辅助读数: 链条逐 token KL(priv||clean) 剖面(前50 token / 其余 / 答案位置),
    验证"KL 权重前移"设计; OCR 题无选项, 只记答案 span 的 logp。

纯 forward 不生成, 405 题 x 3 条件单卡分钟级。prompt 构造 import 自 infer_privilege,
探针效度依赖与正确率实验逐字节一致。投训练前抽 5 题与 verl compute_log_prob 对数校验。

运行 (vision-opd 环境):
  python probe_logprob.py --model Qwen/Qwen3-VL-8B-Instruct --limit 10   # 冒烟
  python probe_logprob.py --model Qwen/Qwen3-VL-8B-Instruct              # 全量
"""
import argparse
import base64
import io
import json
import os
import random
import re

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from infer_privilege import (VOPD_ROOT, TSV_PATH, USER_SUFFIX, TAGS,
                             question_text, BOX_SENTENCE_LABELED, labeled_boxes_str)

ANS_DIR = f"{VOPD_ROOT}/eval/model_answer/treebench"
CONDS = ["clean", "priv", "placebo"]


def placebo_rec(rec, index):
    """真标签 + 确定性随机假坐标 (可复现): 同长度文本, 无真实信息。"""
    rng = random.Random(9000 + index)
    fake = []
    for _ in rec["gt_boxes_norm"]:
        x1, y1 = rng.randint(0, 799), rng.randint(0, 799)
        fake.append([x1, y1, min(1000, x1 + rng.randint(60, 200)),
                     min(1000, y1 + rng.randint(60, 200))])
    return {"gt_boxes_norm": fake, "labels": rec["labels"]}


def build_prompt_text(item, cond, label_rec, index):
    qs = question_text(item)
    if cond == "priv":
        qs += "\n" + BOX_SENTENCE_LABELED.format(boxes=labeled_boxes_str(label_rec))
    elif cond == "placebo":
        qs += "\n" + BOX_SENTENCE_LABELED.format(boxes=labeled_boxes_str(placebo_rec(label_rec, index)))
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{item['image']}"},
        {"type": "text", "text": qs + USER_SUFFIX},
    ]}]


def option_letters(item):
    if item["category"] == "OCR":
        return []
    return re.findall(r"^\s*([A-Z])[\.\)]", str(item["multi-choice options"]), re.M)


def find_answer_token(tokenizer, output, cont_ids):
    """答案字母在 cont_ids 里的 token 位置。BPE 会把 '>C' 合并成单 token, 所以还返回
    该 token 的表面串 (如 '>C'); margin 的候选用同语境替换字母得到 ('>A','>B',...)。"""
    m = re.search(r"<answer>\s*([A-Z])", output)
    if not m:
        return None, None, None
    char_idx = m.start(1)
    enc = tokenizer(output, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc["input_ids"]) != cont_ids.tolist():
        return None, None, None
    for t, (a, b) in enumerate(enc["offset_mapping"]):
        if a <= char_idx < b:
            return t, m.group(1), output[a:b]
    return None, None, None


def forward_logps(model, prompt_inputs, cont_ids, want_dist):
    """teacher-force 续写段。返回 (逐token logp [T] float32-cpu, 可选 [T,V] half-gpu 分布)。"""
    full_ids = torch.cat([prompt_inputs["input_ids"], cont_ids.unsqueeze(0)], dim=1)
    kwargs = {k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask")}
    # processor 的 mm_token_type_ids 只覆盖 prompt 段; 续写段全是文本(type 0), 补零到同长,
    # 否则 M-RoPE 的 get_rope_index 用全长 attention_mask 去索引它会形状崩溃。
    if "mm_token_type_ids" in kwargs:
        pad = torch.zeros((1, cont_ids.shape[0]), dtype=kwargs["mm_token_type_ids"].dtype,
                          device=kwargs["mm_token_type_ids"].device)
        kwargs["mm_token_type_ids"] = torch.cat([kwargs["mm_token_type_ids"], pad], dim=1)
    with torch.inference_mode():
        out = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), **kwargs)
    p0 = prompt_inputs["input_ids"].shape[1]
    logits = out.logits[0, p0 - 1:-1]                      # [T, V] 预测 cont 各 token
    T = cont_ids.shape[0]
    per_tok = torch.empty(T)
    dist = torch.empty((T, logits.shape[-1]), dtype=torch.float16, device=logits.device) if want_dist else None
    for i in range(0, T, 128):
        lp = torch.log_softmax(logits[i:i + 128].float(), dim=-1)
        per_tok[i:i + 128] = lp.gather(-1, cont_ids[i:i + 128, None].to(lp.device)).squeeze(-1).cpu()
        if want_dist is not None and dist is not None:
            dist[i:i + 128] = lp.half()
    return per_tok, dist, logits


def summarize(rows):
    mc = [r for r in rows if all(r[c].get("margin") is not None for c in CONDS)]
    wrong = [r for r in mc if not r["direct_correct"]]
    right = [r for r in mc if r["direct_correct"]]
    print(f"\n===== logprob 探针汇总: {len(rows)} 题, 可算 margin {len(mc)} (错 {len(wrong)} / 对 {len(right)})")

    def dmean(rs, cond):
        v = [r[cond]["margin"] - r["clean"]["margin"] for r in rs]
        return round(sum(v) / len(v), 4) if v else float("nan")

    for cond in ("priv", "placebo"):
        print(f"[{cond}] Δmargin  错题(救回潜力): {dmean(wrong, cond):+}   对题(破坏风险): {dmean(right, cond):+}")
        f_up = sum(1 for r in wrong if r[cond]["tf_letter"] == r["answer"].upper() != r["clean"]["tf_letter"])
        f_dn = sum(1 for r in right if r["clean"]["tf_letter"] == r["answer"].upper() != r[cond]["tf_letter"])
        print(f"[{cond}] 受限argmax翻转  错->对: {f_up}/{len(wrong)}   对->错: {f_dn}/{len(right)}")
    kl = [r["kl"] for r in rows if r.get("kl")]
    if kl:
        f50 = sum(k["first50"] for k in kl) / len(kl)
        rests = [k["rest"] for k in kl if k["rest"] is not None]
        ans = [k["answer_pos"] for k in kl if k["answer_pos"] is not None]
        print(f"KL(priv||clean)/token  前50: {f50:.4f}   其余: "
              f"{sum(rests)/len(rests):.4f}   答案位: {sum(ans)/len(ans):.4f} (n={len(ans)})"
              if rests and ans else f"KL 前50: {f50:.4f}")
    for tag in TAGS:
        g = [r for r in mc if r["category"] == tag and not r["direct_correct"]]
        if g:
            print(f"  错题Δmargin[priv] {tag}: {dmean(g, 'priv'):+} (n={len(g)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--traj-from", default=None, help="direct 轨迹 jsonl; 默认按模型名自动找")
    ap.add_argument("--labels-from", default=f"{VOPD_ROOT}/eval/treebench_gt_labels.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tag = args.model.rstrip("/").split("/")[-1].lower()
    traj_path = args.traj_from
    if traj_path is None:
        for cand in (f"{ANS_DIR}/{tag}-direct-nosys_answer.jsonl", f"{ANS_DIR}/{tag}-direct_answer.jsonl"):
            if os.path.exists(cand):
                traj_path = cand
                break
    if traj_path is None:
        raise SystemExit(f"找不到 {tag} 的 direct 轨迹, 用 --traj-from 指定")
    trajs = {r["index"]: r for r in map(json.loads, open(traj_path))}
    label_map = {r["index"]: r for r in map(json.loads, open(args.labels_from))}
    out_path = args.out or f"{ANS_DIR}/{tag}-logprob-probe.jsonl"
    print(f"trajectories: {traj_path} ({len(trajs)})\nout -> {out_path}")

    df = load_dataset("csv", data_files=TSV_PATH, delimiter="\t")["train"]
    if args.limit:
        df = df.select(range(args.limit))

    done = {}
    if os.path.exists(out_path):
        done = {r["index"]: r for r in map(json.loads, open(out_path))}
        print(f"resume: {len(done)} rows already done")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer

    out_f = open(out_path, "a")
    for item in tqdm(df, desc=f"{tag}-logprob"):
        if item["index"] in done or item["index"] not in trajs:
            continue
        traj = trajs[item["index"]]
        output = traj["output"].split("<|im_end|>")[0]
        if not output.strip():
            continue
        cont_ids = torch.tensor(tokenizer(output, add_special_tokens=False)["input_ids"])
        ans_pos, ans_letter, ans_surface = find_answer_token(tokenizer, output, cont_ids)
        letters = option_letters(item)

        rec = {"index": item["index"], "category": item["category"], "answer": item["answer"],
               "direct_pred": traj["prediction"],
               "direct_correct": traj["prediction"].upper() == item["answer"].upper(),
               "n_cont_tokens": int(cont_ids.shape[0]), "model": args.model,
               "traj_from": os.path.basename(traj_path)}
        dists = {}
        for cond in CONDS:
            messages = build_prompt_text(item, cond, label_map[item["index"]], item["index"])
            image_inputs, _ = process_vision_info(messages)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=image_inputs, padding=True,
                               return_tensors="pt").to(model.device)
            per_tok, dist, logits = forward_logps(model, inputs, cont_ids.to(model.device),
                                                  want_dist=(cond in ("clean", "priv")))
            c = {"mean_logp": round(per_tok.mean().item(), 4),
                 "sum_logp": round(per_tok.sum().item(), 2)}
            if ans_pos is not None and letters and item["answer"].upper() in letters:
                lp_ans = torch.log_softmax(logits[ans_pos].float(), dim=-1)
                lps = {}
                for L in letters:
                    # 同语境候选: 实际 token 表面串里把字母替换 (如 '>C'->'>A'); 须仍是单 token
                    ids = tokenizer(ans_surface.replace(ans_letter, L, 1),
                                    add_special_tokens=False)["input_ids"]
                    if len(ids) == 1:
                        lps[L] = lp_ans[ids[0]].item()
                if item["answer"].upper() in lps and len(lps) > 1:
                    corr = lps[item["answer"].upper()]
                    others = max(v for k, v in lps.items() if k != item["answer"].upper())
                    c["margin"] = round(corr - others, 4)
                    c["letter_logps"] = {k: round(v, 4) for k, v in lps.items()}
                    c["tf_letter"] = max(lps, key=lps.get)
            rec[cond] = c
            if dist is not None:
                dists[cond] = dist
            del logits
        if "clean" in dists and "priv" in dists:
            t, s = dists["priv"], dists["clean"]
            kl = torch.empty(t.shape[0])
            for i in range(0, t.shape[0], 128):
                a, b = t[i:i + 128].float(), s[i:i + 128].float()
                kl[i:i + 128] = (a.exp() * (a - b)).sum(-1).cpu()
            rec["kl"] = {"first50": round(kl[:50].mean().item(), 4),
                         "rest": round(kl[50:].mean().item(), 4) if kl.shape[0] > 50 else None,
                         "answer_pos": round(kl[ans_pos].item(), 4) if ans_pos is not None else None}
        del dists
        done[item["index"]] = rec
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    summarize(list(done.values()))


if __name__ == "__main__":
    main()
