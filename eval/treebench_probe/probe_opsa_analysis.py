"""OPSA 仿写探针第二步 (CPU): 用官方 compute_opsa + 论文口径复刻 Fig2a/3a/3b。

数据: opsa_dump_rows.jsonl (逐位置 lrF/lrH/entF; step1 时 lrF=学生logp, u=lrH-lrF=OPD优势)。

复刻:
  Fig2a  答案 span 优势符号 x 轨迹对错 (他们用 \\boxed{} span; 我们用 gts 新信息词
         在输出中的 token 位置作为答案 span) —— 原始 u 与 去偏 ũ 双口径
  Fig3a  优势分布: 近零(|u|<=1e-4)比例、分桶
  Fig3b  近零比例 x 学生 logp 分位
  官方选择器: compute_opsa(fraction=0.2, entropy模式, batch级) 选中集 x 我们的 gems/ũ

运行: python probe_opsa_analysis.py
"""
import json
import re
import sys
import statistics as st
from collections import defaultdict

import torch

sys.path.insert(0, "/sfs/weka/scratch/nkw3mr/Vision-OPD/TreeVGR/OPSA-code/slime")
from slime.backends.megatron_utils.opsa import compute_opsa  # 官方实现, 纯 torch

from transformers import AutoTokenizer

VOPD = "/sfs/weka/scratch/nkw3mr/Vision-OPD"
ROWS = f"{VOPD}/eval/treebench_probe/opsa_dump_rows.jsonl"
STOP = set("the a an of in on at to for with and or is are was were it its this that side image there".split())


def words(s):
    return set(w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2)


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    rows = [json.loads(l) for l in open(ROWS)]
    print(f"链数: {len(rows)}")

    # ---------- 全局逐词 ū (p^H 加权省略, 实际token口径, 语料级) ----------
    usum, ucnt = defaultdict(float), defaultdict(int)
    for r in rows:
        for v, lf, lh in zip(r["tok"], r["lrF"], r["lrH"]):
            usum[v] += lh - lf; ucnt[v] += 1
    ubar = {v: usum[v] / ucnt[v] for v in usum}

    # ---------- Fig2a: 答案 span 优势 x 对错 ----------
    res = {"raw": defaultdict(list), "deb": defaultdict(list)}
    n_ok = n_bad = 0
    for r in rows:
        ans_new = words(r["gts"]) - words(r["question"])
        if not ans_new:
            continue
        out_w = words(r["output"])
        correct = len(ans_new & out_w) / len(ans_new) >= 0.6
        # 答案 span: 解码后包含答案新信息词的 token 位置
        span = [i for i, v in enumerate(r["tok"])
                if words(tokenizer.decode([v])) & ans_new]
        if not span:
            continue
        u = [r["lrH"][i] - r["lrF"][i] for i in span]
        ud = [r["lrH"][i] - r["lrF"][i] - ubar[r["tok"][i]] for i in span]
        res["raw"][correct].append(st.mean(u))
        res["deb"][correct].append(st.mean(ud))
        n_ok += correct; n_bad += not correct
    print(f"\n== Fig2a 复刻: 答案 span 优势符号 x 对错 (可判 {n_ok+n_bad}, 正确率 {n_ok/(n_ok+n_bad)*100:.1f}%)")
    for tag, lbl in [("raw", "原始 u"), ("deb", "去偏 ũ")]:
        cn = sum(1 for v in res[tag][True] if v < 0)
        ip = sum(1 for v in res[tag][False] if v > 0)
        print(f"  [{lbl}] 正确轨迹span负优势: {cn}/{len(res[tag][True])} = {cn/max(len(res[tag][True]),1)*100:.1f}%"
              f"   错误轨迹span正优势: {ip}/{len(res[tag][False])} = {ip/max(len(res[tag][False]),1)*100:.1f}%"
              f"   总噪声率 {(cn+ip)/(n_ok+n_bad)*100:.1f}%  [论文4B: 20.4/40.8/30.6]")
        print(f"        span均值: 正确 {st.mean(res[tag][True]):+.3f}  错误 {st.mean(res[tag][False]):+.3f}")

    # ---------- Fig3a: 优势分布 ----------
    all_u = [lh - lf for r in rows for lf, lh in zip(r["lrF"], r["lrH"])]
    n = len(all_u)
    print(f"\n== Fig3a 复刻: 优势分布 (N={n})")
    for thr in (1e-4, 1e-2, 0.1):
        print(f"  |u|<={thr}: {sum(1 for v in all_u if abs(v) <= thr)/n*100:.1f}%   [论文 |A|<=1e-4: 51.7%]")
    print(f"  u<0: {sum(1 for v in all_u if v < 0)/n*100:.1f}%  中位 {st.median(all_u):+.3f}")

    # ---------- Fig3b: 近零比例 x 学生 logp 分位 ----------
    pairs = sorted(((lf, abs(lh - lf)) for r in rows for lf, lh in zip(r["lrF"], r["lrH"])),
                   key=lambda x: -x[0])
    print("\n== Fig3b 复刻: |u|<=0.1 比例 x 学生 top-logp 分位  [论文: top20% 内 97.5% 近零]")
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        k = int(frac * n)
        nz = sum(1 for _, au in pairs[:k] if au <= 0.1) / max(k, 1)
        print(f"  top {int(frac*100)}%: {nz*100:.1f}%")

    # ---------- 官方选择器 x 我们的 gems ----------
    log_probs = [torch.tensor(r["lrF"]) for r in rows]
    masks = [torch.ones(len(r["lrF"])) for r in rows]
    ents = [torch.tensor(r["entF"]) for r in rows]
    out = compute_opsa(log_probs, masks, token_fraction=0.2, mode="entropy",
                       entropies=ents, advantage_min=-1.0, advantage_max=-0.5)
    sel_gems_raw = sel_gems_deb = sel_total = 0
    adv_on_sel = []
    for r, sel_mask, adv in zip(rows, out.loss_masks, out.advantages):
        for i in range(len(r["tok"])):
            if sel_mask[i] > 0:
                sel_total += 1
                u = r["lrH"][i] - r["lrF"][i]
                ud = u - ubar[r["tok"][i]]
                sel_gems_raw += u > 1
                sel_gems_deb += ud > 1
                adv_on_sel.append(float(adv[i]))
    print(f"\n== 官方 compute_opsa(0.2, entropy, batch级) 选中 {sel_total} tokens "
          f"({sel_total/n*100:.1f}%), OPSA优势均值 {st.mean(adv_on_sel):.3f}")
    print(f"  选中集内 gems(原始u>+1): {sel_gems_raw/sel_total*100:.1f}%   gems(去偏ũ>+1): {sel_gems_deb/sel_total*100:.1f}%")
    print(f"  -> 官方口径下的错杀率 (OPSA 施负优势而证据教师想抬的 token 比例)")


if __name__ == "__main__":
    main()
