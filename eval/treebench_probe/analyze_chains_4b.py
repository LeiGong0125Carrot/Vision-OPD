"""4B: hide 视角 vs full image (direct) 推理链趋势对比 (9B 那轮分析的 4B 复刻)。

维度: 2x2 翻转矩阵 / 链长 / 词汇分歧(Jaccard) / GT 对象提及(首次位置+频率,
用 treebench_gt_labels.jsonl) / 补偿性推断标记(hedge+不可见) / 感知基底
(外观词 vs 形状结构词) / 修正标记 / 翻转案例摘要。

CPU 即可: python analyze_chains_4b.py
"""
import json
import re
import statistics as st
from collections import Counter, defaultdict

VOPD = "/sfs/weka/scratch/nkw3mr/Vision-OPD"
D = f"{VOPD}/eval/model_answer/treebench"
F_DIRECT = f"{D}/qwen3.5-4b-direct-nothink_answer.jsonl"
F_HIDE = f"{D}/qwen3.5-4b-priv-hide-nothink_answer.jsonl"
F_LABELS = f"{VOPD}/eval/treebench_gt_labels.jsonl"

HEDGE = re.compile(r"\b(likely|probably|appears?|seems?|suggest(?:s|ing)?|might|may be|"
                   r"presumably|possibly|perhaps|assume|infer(?:red)?|guess)\b", re.I)
INVIS = re.compile(r"\b(cannot see|can't see|not visible|hidden|obscured|blacked?[- ]?out|"
                   r"masked|blocked|missing|no visible|hard to see|unclear)\b", re.I)
REVISE = re.compile(r"\b(wait|actually|however|but wait|re-?examine|on second|correction|"
                    r"let me reconsider|hmm)\b", re.I)
APPEAR = re.compile(r"\b(colou?r(?:ed)?|red|blue|green|yellow|white|black|brown|orange|pink|"
                    r"purple|gray|grey|bright|dark|shiny|texture)\b", re.I)
STRUCT = re.compile(r"\b(shape(?:d)?|rectangular|circular|cylindrical|square|round|edge|corner|"
                    r"vertical|horizontal|structure|outline|silhouette|region|position(?:ed)?|"
                    r"located|left side|right side|top|bottom|center|middle)\b", re.I)


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["index"]] = r
    return out


def chain(r):
    o = r["output"].split("<|im_end|>")[0]
    p = o.find("<answer>")
    return (o[:p] if p >= 0 else o).strip()


def toks(s):
    return re.findall(r"[a-z']+", s.lower())


def jacc(a, b):
    A, B = set(a), set(b)
    return len(A & B) / max(len(A | B), 1)


def first_mention_pos(text, names):
    tl = text.lower()
    best = None
    for n in names:
        n = (n or "").strip().lower()
        if len(n) < 3:
            continue
        i = tl.find(n)
        if i >= 0 and (best is None or i < best):
            best = i
    return None if best is None else best / max(len(tl), 1)


def per_kw(rx, words_text, n_words):
    return len(rx.findall(words_text)) / max(n_words, 1) * 100


def main():
    di, hi = load(F_DIRECT), load(F_HIDE)
    labels = {r["index"]: r.get("labels") or [] for r in map(json.loads, open(F_LABELS))}
    idx = sorted(set(di) & set(hi))
    print(f"配对样本: {len(idx)}")

    groups = defaultdict(list)   # 2x2
    bycat = defaultdict(Counter)
    rows = []
    for i in idx:
        d, h = di[i], hi[i]
        dc = d["prediction"].strip().upper()[:1] == d["answer"].strip().upper()[:1]
        hc = h["prediction"].strip().upper()[:1] == h["answer"].strip().upper()[:1]
        g = ("A_both_ok" if dc and hc else "B_hide_rescue" if not dc and hc
             else "C_hide_harm" if dc and not hc else "D_both_wrong")
        cat = d["category"].split("/")[-1]
        groups[g].append(i)
        bycat[cat][g] += 1
        cd, ch = chain(d), chain(h)
        td, th = toks(cd), toks(ch)
        rows.append(dict(
            i=i, g=g, cat=cat,
            len_d=len(td), len_h=len(th),
            jacc=jacc(td, th),
            jacc300=jacc(toks(cd[:300]), toks(ch[:300])),
            fm_d=first_mention_pos(cd, labels.get(i, [])),
            fm_h=first_mention_pos(ch, labels.get(i, [])),
            hedge_d=per_kw(HEDGE, cd, len(td)), hedge_h=per_kw(HEDGE, ch, len(th)),
            invis_d=per_kw(INVIS, cd, len(td)), invis_h=per_kw(INVIS, ch, len(th)),
            rev_d=per_kw(REVISE, cd, len(td)), rev_h=per_kw(REVISE, ch, len(th)),
            app_d=per_kw(APPEAR, cd, len(td)), app_h=per_kw(APPEAR, ch, len(th)),
            str_d=per_kw(STRUCT, cd, len(td)), str_h=per_kw(STRUCT, ch, len(th)),
        ))

    acc_d = sum(1 for r in rows if r["g"] in ("A_both_ok", "C_hide_harm")) / len(rows) * 100
    acc_h = sum(1 for r in rows if r["g"] in ("A_both_ok", "B_hide_rescue")) / len(rows) * 100
    print(f"\n== 总体: direct {acc_d:.2f}%  hide {acc_h:.2f}%  (Δ {acc_h-acc_d:+.2f})")
    print("== 2x2 翻转矩阵:")
    for g in ("A_both_ok", "B_hide_rescue", "C_hide_harm", "D_both_wrong"):
        print(f"  {g:<14} {len(groups[g]):>4}  ({len(groups[g])/len(rows)*100:.1f}%)")
    print("\n== 分类别 (n / B_rescue / C_harm / D_both_wrong):")
    for cat, c in sorted(bycat.items(), key=lambda x: -sum(x[1].values())):
        n = sum(c.values())
        print(f"  {cat:<22} n={n:<4} B={c['B_hide_rescue']:<3} C={c['C_hide_harm']:<3} D={c['D_both_wrong']}")

    def agg(key, sub=None):
        v = [r[key] for r in (sub or rows) if r[key] is not None]
        return st.mean(v) if v else float("nan")

    print(f"\n== 链长 (词数): direct {agg('len_d'):.0f}  hide {agg('len_h'):.0f}")
    for g in ("A_both_ok", "B_hide_rescue", "C_hide_harm", "D_both_wrong"):
        sub = [r for r in rows if r["g"] == g]
        print(f"  {g:<14} direct {agg('len_d', sub):>5.0f}  hide {agg('len_h', sub):>5.0f}")

    print(f"\n== 词汇分歧: 整链 Jaccard {agg('jacc'):.3f}   前300字 {agg('jacc300'):.3f}")
    for g in ("A_both_ok", "B_hide_rescue", "C_hide_harm", "D_both_wrong"):
        sub = [r for r in rows if r["g"] == g]
        print(f"  {g:<14} 整链 {agg('jacc', sub):.3f}  前300字 {agg('jacc300', sub):.3f}")

    nd = sum(1 for r in rows if r["fm_d"] is not None)
    nh = sum(1 for r in rows if r["fm_h"] is not None)
    print(f"\n== GT 对象提及: 提及率 direct {nd/len(rows)*100:.1f}%  hide {nh/len(rows)*100:.1f}%")
    print(f"   首次提及位置(0=开头,1=结尾): direct {agg('fm_d'):.3f}  hide {agg('fm_h'):.3f}")

    print("\n== 每百词标记频率 (direct -> hide):")
    for name, kd, kh in [("hedge 推断词", "hedge_d", "hedge_h"), ("不可见/遮挡", "invis_d", "invis_h"),
                          ("修正词", "rev_d", "rev_h"), ("外观/颜色词", "app_d", "app_h"),
                          ("形状/结构/位置词", "str_d", "str_h")]:
        print(f"  {name:<14} {agg(kd):.2f} -> {agg(kh):.2f}   (Δ {agg(kh)-agg(kd):+.2f})")
        for g in ("B_hide_rescue", "C_hide_harm", "D_both_wrong"):
            sub = [r for r in rows if r["g"] == g]
            print(f"     {g:<14} {agg(kd, sub):.2f} -> {agg(kh, sub):.2f}")

    print("\n== 翻转案例 (各3条, index/类别/链首150字):")
    for g in ("B_hide_rescue", "C_hide_harm"):
        print(f"  --- {g}")
        for i in groups[g][:3]:
            print(f"  [{i}] {di[i]['category'].split('/')[-1]}  d:{di[i]['prediction'][:1]} h:{hi[i]['prediction'][:1]} gt:{di[i]['answer']}")
            print(f"      D: {chain(di[i])[:150]!r}")
            print(f"      H: {chain(hi[i])[:150]!r}")

    json.dump(rows, open(f"{VOPD}/eval/treebench_probe/chains4b_rows.json", "w"))
    print(f"\nrows -> eval/treebench_probe/chains4b_rows.json")


if __name__ == "__main__":
    main()
