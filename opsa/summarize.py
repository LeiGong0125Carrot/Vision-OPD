"""CPU 汇总: 读各 probe 的 rows.jsonl 出报告。

  python summarize.py --source sharpness --rows results/sharpness_rows_qwen3.5-4b.jsonl
  python summarize.py --source sharpness --rows results/sharpness_rows_qwen3-1.7b_aime.jsonl
  python summarize.py --source answer_noise --rows results/answer_noise_rows_qwen3.5-4b.jsonl
  python summarize.py --source opsa_dump      # 交叉验证: 已有 RL-37K T=1.0 dump (4B)

sharpness 四组统计 (对应讨论中的 a/b/c/d):
  (a) 全序列熵分布 + top1>0.9 / H>1 占比
  (b) lowest-20%-logp token 的熵分布 (per-chain 与全池两种口径)
  (c) 高熵位置 (H>1) top-5 候选反思词命中率
  (d) tail-20% token 构成: 坐标数字 / 普通数字 / 答案字母 / 标点 / 其他 + 高频表
"""
import argparse
import json
import os
import re

OPSA_DIR = os.path.dirname(os.path.abspath(__file__))
VOPD_ROOT = os.environ.get("VOPD_ROOT", os.path.dirname(OPSA_DIR))

REFLECT = {"wait", "however", "but", "alternatively", "hmm", "perhaps", "check", "might",
           "actually", "re-examine", "reexamine", "reconsider", "correction", "hold",
           "second", "double-check", "let", "maybe"}
COORD_RE = re.compile(r"\[\s*\d[\d,\s]*\]")


def q(vals, ps=(10, 25, 50, 75, 90, 99)):
    if not vals:
        return "n=0"
    s = sorted(vals)
    parts = [f"p{p}={s[min(len(s)-1, int(len(s)*p/100))]:.3f}" for p in ps]
    return f"mean={sum(vals)/len(vals):.3f} " + " ".join(parts)


def norm_word(s):
    return s.strip().lower().strip(".,!?:;\"'()")


def tail_mask_per_chain(logp, frac=0.2):
    T = len(logp)
    k = max(1, int(frac * T))
    order = sorted(range(T), key=lambda i: logp[i])
    m = [False] * T
    for i in order[:k]:
        m[i] = True
    return m


def load_rows(path, limit=0):
    rows = []
    for line in open(path):
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def get_tokenizer(rows, model_arg):
    from transformers import AutoTokenizer
    model = model_arg or rows[0].get("model")
    print(f"[tokenizer: {model}]")
    return AutoTokenizer.from_pretrained(model)


def reconstruct(tokenizer, tok):
    pieces = [tokenizer.decode([t], skip_special_tokens=False,
                               clean_up_tokenization_spaces=False) for t in tok]
    offs, off = [], 0
    for p in pieces:
        offs.append((off, off + len(p)))
        off += len(p)
    return pieces, offs, "".join(pieces)


def summarize_sharpness(rows, tokenizer, has_top5=True):
    all_ent, all_logp, top1_hi, hi_ent = [], [], 0, 0
    tail_ent_chain, n_tok = [], 0
    # (c)
    hi_pos = refl_top5 = refl_sampled = 0
    # (d)
    from collections import Counter
    comp = Counter()
    tail_freq = Counter()
    # 全池 tail 需要两遍; 先收集 (logp, ent)
    pool = []

    for r in rows:
        logp, ent, tok = r["logp" if "logp" in r else "lrF"], r["ent" if "ent" in r else "entF"], r["tok"]
        T = len(logp)
        n_tok += T
        all_ent.extend(ent)
        all_logp.extend(logp)
        pool.extend(zip(logp, ent))
        if has_top5:
            top1_hi += sum(1 for row in r["top5_p"] if row[0] > 0.9)
        hi_ent += sum(1 for e in ent if e > 1.0)

        tmask = tail_mask_per_chain(logp)
        tail_ent_chain.extend(e for e, m in zip(ent, tmask) if m)

        pieces, offs, text = reconstruct(tokenizer, tok)
        # 坐标 span
        coord_ranges = [(m.start(), m.end()) for m in COORD_RE.finditer(text)]
        ans_m = re.search(r"<answer>\s*([A-Z])", text)
        ans_char = ans_m.start(1) if ans_m else -1

        for i in range(T):
            if has_top5 and ent[i] > 1.0:
                hi_pos += 1
                cands = [norm_word(tokenizer.decode([c])) for c in r["top5_ids"][i]]
                if any(c in REFLECT for c in cands):
                    refl_top5 += 1
                if norm_word(pieces[i]) in REFLECT:
                    refl_sampled += 1
            if tmask[i]:
                s = pieces[i]
                ss = s.strip()
                a, b = offs[i]
                in_coord = any(ca <= a < cb for ca, cb in coord_ranges)
                is_ans = ans_char >= 0 and a <= ans_char < b
                if is_ans:
                    comp["answer_letter"] += 1
                elif ss and all(ch in "0123456789," for ch in ss):
                    comp["coord_digit" if in_coord else "digit"] += 1
                elif not ss or all(not ch.isalnum() for ch in ss):
                    comp["punct_ws"] += 1
                else:
                    comp["word"] += 1
                tail_freq[repr(s)] += 1

    print(f"\n===== 尖锐度报告: {len(rows)} chains, {n_tok} tokens")
    print(f"(a) 全序列熵     {q(all_ent)}")
    print(f"    全序列logp   {q(all_logp)}")
    if has_top5:
        print(f"    frac(top1>0.9) = {top1_hi/max(n_tok,1)*100:.1f}%")
    print(f"    frac(H>1)      = {hi_ent/max(n_tok,1)*100:.1f}%")

    print(f"(b) tail-20% 熵 (per-chain 口径)  {q(tail_ent_chain)}")
    pool.sort(key=lambda x: x[0])
    gk = max(1, int(0.2 * len(pool)))
    gtail = [e for _, e in pool[:gk]]
    print(f"    tail-20% 熵 (全池口径)        {q(gtail)}")
    lo = sum(1 for e in tail_ent_chain if e < 0.1) / max(len(tail_ent_chain), 1)
    hi = sum(1 for e in tail_ent_chain if e > 1.0) / max(len(tail_ent_chain), 1)
    print(f"    tail 内: 低熵(H<0.1,确信但低logp)={lo*100:.1f}%  高熵(H>1,真不确定)={hi*100:.1f}%")

    if has_top5:
        print(f"(c) 高熵位置 (H>1): {hi_pos};  top-5 含反思词 = "
              f"{refl_top5/max(hi_pos,1)*100:.1f}%;  采到的即反思词 = {refl_sampled/max(hi_pos,1)*100:.1f}%")

    tot = sum(comp.values())
    print(f"(d) tail-20% 构成 (n={tot}): " + "  ".join(
        f"{k}={v/max(tot,1)*100:.1f}%" for k, v in comp.most_common()))
    print("    tail 高频 token:")
    for s, c in tail_freq.most_common(30):
        print(f"      {s:>16}  n={c}")


def summarize_answer_noise(rows):
    n_ans = 0
    noisy_cor = noisy_inc = n_cor = n_inc = 0
    u_ans_cor, u_ans_inc = [], []
    nz_total = nz_count = 0
    # Fig 3b: 按学生 logp 从高到低保留 top-X%, 其中 |u|<=1e-4 的占比
    pool = []
    tail_stats = {"gems": 0, "neutral": 0, "confirm": 0, "n": 0}

    for r in rows:
        u = [h - f for h, f in zip(r["lrH"], r["lrF"])]
        for f, uu in zip(r["lrF"], u):
            pool.append((f, abs(uu) <= 1e-4))
        nz_total += len(u)
        nz_count += sum(1 for uu in u if abs(uu) <= 1e-4)
        tmask = tail_mask_per_chain(r["lrF"])
        for uu, m in zip(u, tmask):
            if m:
                tail_stats["n"] += 1
                if uu > 1:
                    tail_stats["gems"] += 1
                elif uu < -1:
                    tail_stats["confirm"] += 1
                else:
                    tail_stats["neutral"] += 1
        ap = r.get("answer_pos")
        if ap is None or ap >= len(u):
            continue
        n_ans += 1
        ua = u[ap]
        if r["correct"]:
            n_cor += 1
            u_ans_cor.append(ua)
            if ua < 0:
                noisy_cor += 1
        else:
            n_inc += 1
            u_ans_inc.append(ua)
            if ua > 0:
                noisy_inc += 1

    print(f"\n===== 答案 token 噪声报告: {len(rows)} chains, 有答案 token 的 {n_ans}")
    print(f"论文对照 (Qwen3-4B teacher, 数学域): 正确链负优势 20.4%, 错误链正优势 40.8%, 总噪声 30.6%")
    if n_cor:
        print(f"正确链 (n={n_cor}): 答案 token u<0 (噪声) = {noisy_cor/n_cor*100:.1f}%   "
              f"mean u = {sum(u_ans_cor)/n_cor:+.3f}")
    if n_inc:
        print(f"错误链 (n={n_inc}): 答案 token u>0 (噪声) = {noisy_inc/n_inc*100:.1f}%   "
              f"mean u = {sum(u_ans_inc)/n_inc:+.3f}")
    if n_ans:
        print(f"总噪声率 = {(noisy_cor+noisy_inc)/n_ans*100:.1f}%")

    print(f"\n§3.1 复现: frac(|u|<=1e-4) 全 token = {nz_count/max(nz_total,1)*100:.1f}%  (论文 51.7%)")
    pool.sort(key=lambda x: -x[0])  # 学生 logp 从高到低
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        k = max(1, int(frac * len(pool)))
        nz = sum(1 for _, z in pool[:k] if z) / k
        print(f"    保留 top-{int(frac*100)}% 高logp token: 近零占比 = {nz*100:.1f}%"
              + ("  (论文: 97.5%)" if frac == 0.2 else "")
              + ("  (论文: 96.6%)" if frac == 0.4 else "")
              + ("  (论文: 51.7%)" if frac == 1.0 else ""))
    n = max(tail_stats["n"], 1)
    print(f"\ntail-20% 上教师意见: gems(u>+1)={tail_stats['gems']/n*100:.1f}%  "
          f"neutral={tail_stats['neutral']/n*100:.1f}%  confirm(u<-1)={tail_stats['confirm']/n*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["sharpness", "answer_noise", "opsa_dump"])
    ap.add_argument("--rows", default=None)
    ap.add_argument("--model", default=None, help="tokenizer 来源 (默认取 rows 的 model 字段)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.source == "opsa_dump":
        path = args.rows or f"{VOPD_ROOT}/eval/treebench_probe/opsa_dump_rows.jsonl"
        rows = load_rows(path, args.limit)
        print(f"[交叉验证: RL-37K T=1.0 dump, {len(rows)} chains; 无 top5 -> 跳过 (c)/top1 统计]")
        tokenizer = get_tokenizer(rows, args.model or "Qwen/Qwen3.5-4B")
        summarize_sharpness(rows, tokenizer, has_top5=False)
        return

    path = args.rows if os.path.isabs(args.rows) else os.path.join(OPSA_DIR, args.rows)
    rows = load_rows(path, args.limit)
    print(f"loaded {len(rows)} rows from {path}")
    if args.source == "sharpness":
        tokenizer = get_tokenizer(rows, args.model)
        summarize_sharpness(rows, tokenizer, has_top5=True)
    else:
        summarize_answer_noise(rows)


if __name__ == "__main__":
    main()
