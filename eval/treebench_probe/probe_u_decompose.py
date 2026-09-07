"""u = log p^H − log p^F 的 DC/局域分解探针(训练分布,region vs full,冻结 4B 教师)。

问题: SA 重建目标 q = softmax(log p^H + β·u) 在训练中表现出"风格累积"伤害
(hedge/巡视被系统性剥除)。假设 u 可分解为:
  - 口味/DC 分量: 同一词在整条链所有位置被同向推压(视图诱导的语域偏移, 稠密)
  - 证据分量: 只在个别位置冒尖的局域修正(稀疏)
判据: 对每个词 v, ū(v) = 该词在链内所有出现位置的 u 均值;残差 ũ_t(v)=u_t(v)−ū(v)。
DC 能量占比 = 1 − Σũ²/Σu²。占比高 ⇒ 口味主导 ⇒ 去均值修复有据。

数据: rollouts/SA-OPD-.../1.jsonl(step1 = 初始策略链, 即训练第一步 loss 见到的人群),
按 question 文本回链 train_sa4k.parquet 取 region crops 与原图。
支撑: F 视图 top-100(初始时 p^S≡p^F, 即学生支撑)。

运行(单卡 ~15min):
  python probe_u_decompose.py --n 50
"""
import argparse
import json
import os
import re
from collections import defaultdict

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

VOPD = "/sfs/weka/scratch/nkw3mr/Vision-OPD"
PARQUET = f"{VOPD}/data/TreeVGR-RL-37K/train_sa4k.parquet"
DUMP = f"{VOPD}/rollouts/SA-OPD-Qwen3.5-4B/1.jsonl"
OUT = f"{VOPD}/eval/treebench_probe/u_decompose_rows.jsonl"
MAX_POS = 256
TOPK = 100


def build_inputs(processor, model, contents, forced_text):
    messages = [{"role": "user", "content": contents}]
    image_inputs, _ = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    inputs = processor(text=[text], images=image_inputs, padding=True,
                       return_tensors="pt").to(model.device)
    return inputs


def forced_logits(model, inputs, cont_ids):
    full_ids = torch.cat([inputs["input_ids"], cont_ids.unsqueeze(0).to(model.device)], dim=1)
    kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    if "mm_token_type_ids" in kwargs:
        pad = torch.zeros((1, cont_ids.shape[0]), dtype=kwargs["mm_token_type_ids"].dtype,
                          device=kwargs["mm_token_type_ids"].device)
        kwargs["mm_token_type_ids"] = torch.cat([kwargs["mm_token_type_ids"], pad], dim=1)
    with torch.inference_mode():
        out = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), **kwargs)
    p0 = inputs["input_ids"].shape[1]
    return out.logits[0, p0 - 1: p0 - 1 + cont_ids.shape[0]].float()


def img_content(path):
    return {"type": "image", "image": path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n", type=int, default=50, help="每个 step dump 抽取上限")
    ap.add_argument("--steps", default="1", help="如 '1' 或 '1,14,28,41' 或 'all'")
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    qmap = {}
    for _, row in df.iterrows():
        q = str(row["extra_info"]["question"]).strip()
        qmap.setdefault(q, row)

    import glob
    dump_dir = os.path.dirname(DUMP)
    if args.steps == "all":
        dumps = sorted(glob.glob(f"{dump_dir}/*.jsonl"), key=lambda p: int(os.path.basename(p)[:-6]))
    else:
        dumps = [f"{dump_dir}/{s.strip()}.jsonl" for s in args.steps.split(",")]

    samples = []
    seen_q = set()
    for dp in dumps:
        step = int(os.path.basename(dp)[:-6])
        cnt = 0
        for line in open(dp):
            r = json.loads(line)
            m = re.search(r"user\n\n?(.*?)\nassistant", r["input"], re.S)
            q = (m.group(1).strip() if m else "")
            if q in qmap and q not in seen_q and len(r["output"].split()) >= 10:
                samples.append((qmap[q], r["output"], step))
                seen_q.add(q)
                cnt += 1
            if cnt >= args.n:
                break
    print(f"matched samples: {len(samples)} (steps: {sorted(set(s for _,_,s in samples))[:8]}...)")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer

    # 全局逐词聚合 (跨序列): 用于风格词表
    glob_sum = defaultdict(float)
    glob_cnt = defaultdict(int)
    rows = []
    out_f = open(OUT, "w")
    for row, output, step in tqdm(samples, desc="u-decompose"):
        question = str(row["extra_info"]["question"]).strip()
        cont_ids = torch.tensor(tokenizer(output, add_special_tokens=False)["input_ids"][:MAX_POS])
        if cont_ids.numel() < 8:
            continue
        img_path = row["images"][0]["path"]
        teacher_paths = [b["path"] for b in row["bbox_images"]]

        in_F = build_inputs(processor, model,
                            [img_content(img_path), {"type": "text", "text": question}], output)
        in_H = build_inputs(processor, model,
                            [*[img_content(p) for p in teacher_paths], {"type": "text", "text": question}], output)
        lg_F = forced_logits(model, in_F, cont_ids)
        lg_H = forced_logits(model, in_H, cont_ids)
        lp_F = torch.log_softmax(lg_F, -1)
        lp_H = torch.log_softmax(lg_H, -1)

        # ---- 长度无关的配对指标 (同链, H/F 双视图; 逐位置量再取均值) ----
        eos_id = tokenizer.eos_token_id
        realized = cont_ids.to(lp_F.device)
        lr_H = lp_H.gather(-1, realized.unsqueeze(-1)).squeeze(-1)   # 实际 token 的 logp
        lr_F = lp_F.gather(-1, realized.unsqueeze(-1)).squeeze(-1)
        ent_H = -(lp_H.exp() * lp_H).sum(-1)                         # 逐位置熵 (nats)
        ent_F = -(lp_F.exp() * lp_F).sum(-1)
        eos_pH = lp_H[:, eos_id].exp()                               # 逐位置停机概率
        eos_pF = lp_F[:, eos_id].exp()
        paired = dict(
            d_logp_tok=float((lr_H - lr_F).mean()),      # H 相对 F 每 token 更喜欢这条链多少
            ent_H=float(ent_H.mean()), ent_F=float(ent_F.mean()),
            d_ent=float((ent_H - ent_F).mean()),         # 负 = H 视图更锐
            eos_hazard_H=float(eos_pH.mean()), eos_hazard_F=float(eos_pF.mean()),
        )

        idx = lp_F.topk(TOPK, -1).indices                     # [T,K] 支撑 = F(≈学生)top-100
        uF = lp_F.gather(-1, idx)
        uH = lp_H.gather(-1, idx)
        idx = idx.cpu()
        u = (uH - uF).cpu()                                    # [T,K] 分解阶段走 CPU
        pH = uH.exp().cpu()

        # 逐词 DC: ū(v) = 链内该词出现位置的均值
        T, K = u.shape
        flat_v = idx.reshape(-1)
        flat_u = u.reshape(-1)
        sums = defaultdict(float); cnts = defaultdict(int)
        for v, val in zip(flat_v.tolist(), flat_u.tolist()):
            sums[v] += val; cnts[v] += 1
        ubar = {v: sums[v] / cnts[v] for v in sums}
        ubar_t = torch.tensor([[ubar[v] for v in r_] for r_ in idx.tolist()])
        resid = u - ubar_t

        e_tot = float((u ** 2).sum())
        e_res = float((resid ** 2).sum())
        w = pH
        e_tot_w = float((w * u ** 2).sum())
        e_res_w = float((w * resid ** 2).sum())
        # 残差能量的位置集中度: top5% 位置占多少残差能量
        per_pos = (resid ** 2).sum(-1)
        k5 = max(1, int(0.05 * T))
        conc = float(per_pos.topk(k5).values.sum() / per_pos.sum().clamp_min(1e-9))

        eos_id = tokenizer.eos_token_id
        eos_u = ubar.get(eos_id, None)

        for v in sums:
            glob_sum[v] += sums[v]; glob_cnt[v] += cnts[v]

        rec = dict(orig_row=int(row["extra_info"]["orig_row"]), T=T, step=step,
                   dc_share=1 - e_res / max(e_tot, 1e-9),
                   dc_share_pHweighted=1 - e_res_w / max(e_tot_w, 1e-9),
                   resid_top5pos_share=conc,
                   eos_ubar=eos_u, **paired)
        rows.append(rec)
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
    out_f.close()

    import statistics as st
    print(f"\n===== u 分解: {len(rows)} 条链")
    for k, lbl in [("dc_share", "DC(口味)能量占比"), ("dc_share_pHweighted", "DC占比(p^H加权)"),
                   ("resid_top5pos_share", "残差能量落在top5%位置的份额")]:
        v = [r[k] for r in rows]
        print(f"  {lbl}: 中位 {st.median(v):.3f}  均值 {st.mean(v):.3f}")
    ev = [r["eos_ubar"] for r in rows if r["eos_ubar"] is not None]
    if ev:
        print(f"  EOS 的 ū: 均值 {st.mean(ev):+.3f} (正 = H 视图整体想更早结束 = 压缩机制)")

    print("\n  配对指标 (同链 H/F 双视图, 逐位置均值 -> 长度无关):")
    print(f"    Δlogp/token (H−F, 实际token): 中位 {st.median([r['d_logp_tok'] for r in rows]):+.3f}")
    print(f"    熵: H视图 {st.median([r['ent_H'] for r in rows]):.3f}  F视图 {st.median([r['ent_F'] for r in rows]):.3f}  "
          f"Δ中位 {st.median([r['d_ent'] for r in rows]):+.3f} (负=H更锐)")
    print(f"    EOS 逐位置停机概率: H {st.median([r['eos_hazard_H'] for r in rows]):.4f}  "
          f"F {st.median([r['eos_hazard_F'] for r in rows]):.4f}  "
          f"比值中位 {st.median([r['eos_hazard_H']/max(r['eos_hazard_F'],1e-9) for r in rows]):.2f}x")

    # 按训练阶段分段 (注意: 各 step 题目集不相交, 组成混淆未消; 配对差分已消掉题内变量)
    buckets = [(1, 5), (6, 15), (16, 30), (31, 41)]
    if len(set(r["step"] for r in rows)) > 3:
        print("\n  按训练阶段 (配对口径):")
        for lo, hi in buckets:
            sub = [r for r in rows if lo <= r["step"] <= hi]
            if len(sub) < 10:
                continue
            hz = st.median([r['eos_hazard_H']/max(r['eos_hazard_F'],1e-9) for r in sub])
            print(f"    step {lo:>2}-{hi:<2} (n={len(sub):>3}): "
                  f"Δlogp/tok {st.median([r['d_logp_tok'] for r in sub]):+.3f}  "
                  f"Δ熵 {st.median([r['d_ent'] for r in sub]):+.3f}  "
                  f"EOS风险比 {hz:.2f}x  DCw {st.median([r['dc_share_pHweighted'] for r in sub]):.3f}")
    print("\n  全局风格词表 (|ū| 最大, 出现>=30 次):")
    cand = [(v, glob_sum[v] / glob_cnt[v], glob_cnt[v]) for v in glob_sum if glob_cnt[v] >= 30]
    cand.sort(key=lambda x: -abs(x[1]))
    for v, m, c in cand[:24]:
        print(f"    {tokenizer.decode([v])!r:>16}  ū={m:+.3f}  n={c}")


if __name__ == "__main__":
    main()
