"""TreeBench 官方口径判卷: 先首字母规则, 规则判不出的送 LLM judge (与 V* 判卷同款 prompt)。

用法 (需先起 judge 服务, 如 gpt-oss-120b):
  python judge_treebench.py --models qwen3.5-4b-direct-nothink sa-opd-...-step14-priv-none-nothink \
      --api-base http://localhost:8813/v1/ --judge-model openai/gpt-oss-120b
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from openai import OpenAI

from infer_privilege import VOPD_ROOT, TSV_PATH

ANS_DIR = f"{VOPD_ROOT}/eval/model_answer/treebench"
OUT_DIR = f"{VOPD_ROOT}/eval/judge/treebench"

PROMPT = ("Your task is to judge whether the response expresses the same meaning as the answer "
          "of a question.\nThe question is: {q}\nThe answer is: {a}\nThe response is: {r}\n"
          "Please check and compare them and then judge. If the response is correct, your output "
          "should be Yes. Otherwise, your output should be No. Directly give me your output.")


def gt_text(item):
    """正确答案的完整文本: 字母 + 选项内容 (OCR 无选项则用 answer 原文)。"""
    ans = str(item["answer"]).strip()
    opts = str(item.get("multi-choice options") or "")
    m = re.search(rf"^\s*\(?{re.escape(ans)}[\.\)]\s*(.*)$", opts, re.M)
    return f"({ans}) {m.group(1).strip()}" if m else ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--api-base", default="http://localhost:8813/v1/")
    ap.add_argument("--judge-model", default="openai/gpt-oss-120b")
    args = ap.parse_args()

    df = load_dataset("csv", data_files=TSV_PATH, delimiter="\t")["train"]
    items = {it["index"]: it for it in df}
    client = OpenAI(api_key="EMPTY", base_url=args.api_base)
    os.makedirs(OUT_DIR, exist_ok=True)

    def llm_judge(question, answer_text, response):
        try:
            r = client.chat.completions.create(
                model=args.judge_model,
                messages=[{"role": "user", "content": PROMPT.format(q=question, a=answer_text,
                                                                    r=response[:2000])}],
                temperature=0, max_tokens=2048)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            return f"ERROR:{e}"

    summary = []
    for model in args.models:
        path = f"{ANS_DIR}/{model}_answer.jsonl"
        recs = [json.loads(l) for l in open(path)]
        rule_yes, pending = 0, []
        for r in recs:
            pred = str(r.get("prediction", "")).strip().upper()[:1]
            ans = str(r.get("answer", "")).strip().upper()[:1]
            if pred and pred == ans:
                r["judge"], r["judge_source"] = "Yes", "first letter"
                rule_yes += 1
            else:
                pending.append(r)
        def run_one(r):
            it = items[r["index"]]
            out = r["output"].split("<|im_end|>")[0]
            q = str(it["question"]) + "\n" + str(it.get("multi-choice options") or "")
            r["judge"] = llm_judge(q, gt_text(it), out)
            r["judge_source"] = "llm"
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(run_one, pending))
        llm_yes = sum(1 for r in pending if re.match(r"^yes\b", r["judge"], re.I))
        n = len(recs)
        total = rule_yes + llm_yes
        with open(f"{OUT_DIR}/{model}_answer.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        errs = sum(1 for r in pending if str(r["judge"]).startswith("ERROR"))
        summary.append((model, total / n * 100, rule_yes / n * 100, llm_yes, len(pending), errs))
        print(f"{model}: judge口径 {total}/{n} = {total/n*100:.2f}%  "
              f"(规则 {rule_yes}, LLM救回 {llm_yes}/{len(pending)}, 错误 {errs})")

    print(f"\n{'模型':<52}{'judge口径':>9}{'规则下界':>9}{'LLM救回':>8}")
    for m, j, rb, ly, np_, e in summary:
        print(f"{m:<52}{j:>8.2f}%{rb:>8.2f}%{ly:>5}/{np_}")


if __name__ == "__main__":
    main()
