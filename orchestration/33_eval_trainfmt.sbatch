#!/bin/bash
# V* 评测 —— 用**训练同款 prompt 格式**重测(bbox-then-answer 范式被真正触发)。
#
# 动机: 32 号作业用的是 V* 裸 prompt(问题+选项, 无格式指令), 而训练时每道题
# 都带 'Respond with exactly one JSON object: {"bbox_2d": ..., "answer": ...}'。
# 也就是说 bbox->answer 通路在那次评测里根本没被触发, trained model 是在
# 一个没见过的裸问答分布上被测的 —— step260 掉的 9.4 个点有两种解释:
#   (a) 通用能力真回退了;  (b) 评测 OOD, 范式没伤模型是我们测歪了。
# 本作业补上 2x2 的另一半: 同样 4 个模型 x 训练格式 prompt。
#   - trained + 训练格式 恢复到 base 水平 -> 支持 (b)
#   - 照样掉                              -> 坐实 (a)
#   - base + 训练格式 这一格顺带测出指令本身对未训模型是增益还是干扰
#
# prompt 改写完全对齐训练分布(见 data/train_xy_rb.parquet 的 prompt 列):
#   问题 + 空行 + "A. xxx" 风格选项(V* 原本是 "(A) xxx") + 空行 + 指令原文
#
#SBATCH --job-name=vopd-eval-fmt
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/evalfmt-%j.log
#SBATCH --open-mode=append

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
mkdir -p "$SETUP/logs"

TAG="${TAG:-vopd-xy_rb-e1}"
MERGE_ROOT="$VOPD_ROOT/merged"
export PORT="${PORT:-8811}"
BJSON="$SETUP/vstar_trainfmt.json"

echo "=========================================================="
echo " V* 训练格式评测   $(date)"
echo " 节点=$(hostname)  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================================="

# ---------- 1. 生成训练格式版 vstar.json ----------
# 指令原文逐字取自 train_xy_rb.parquet, 不手抄 —— 手抄错一个字就不是同分布了。
"$VOPD_PY" - "$BJSON" <<'PY'
import json, re, sys
import pandas as pd
df = pd.read_parquet("/sfs/weka/scratch/nkw3mr/Vision-OPD/data/train_xy_rb.parquet")
c = df.iloc[0]["prompt"][0]["content"]
i = c.find("Respond with exactly one JSON object")
assert i > 0, "训练 parquet 里找不到格式指令"
INSTR = c[i:]
print("指令原文:", repr(INSTR))

src = json.load(open("/sfs/weka/scratch/nkw3mr/Vision-OPD/eval/vstar.json"))
OPT = re.compile(r"^\(([A-D])\)\s*(.*)$")
# V* 每条 query 末尾自带 "Answer with the option's letter ..." 指令(191/191 条,
# 已实测确认) —— 必须剥掉, 否则会和 JSON 指令打架。注意: 这意味着 32 号作业的
# "裸 prompt" 其实带着这条直接作答指令, 写结论时要说清楚。
TAIL = "Answer with the option"
out = []
for r in src:
    lines = [l for l in r["query"].strip().splitlines() if not l.strip().startswith(TAIL)]
    q, opts = [], []
    for l in lines:
        m = OPT.match(l.strip())
        (opts if m else q).append(f"{m.group(1)}. {m.group(2)}" if m else l)
    # V* 有 84/191 道二选一题(只有 A/B), 不能断言恒为 4
    assert 2 <= len(opts) <= 4, f"qid={r['question_id']} 选项数 {len(opts)}"
    assert any(x.strip() for x in q), f"qid={r['question_id']} 问题为空"
    r2 = dict(r)
    r2["query"] = "\n".join(q).strip() + "\n\n" + "\n".join(opts) + "\n\n" + INSTR
    out.append(r2)
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print(f"写出 {len(out)} 条 -> {sys.argv[1]}")
print("样例 query:")
print(out[0]["query"])
PY
rc=$?; [ $rc -ne 0 ] && { echo "❌ 生成 benchmark json 失败"; exit 1; }

# ---------- 2. 逐个模型: 起服务 -> infer -> 停 ----------
# 不走 run_eval.sh(它的 BENCHMARK_JSON_MAP 写死在仓库里, 加自定义 json 得改仓库),
# 直接调 infer.py。输出 tag 带 -trainfmt 后缀, 和 32 号作业的裸 prompt 结果并排落在
# model_answer/vstar/ 下, 方便 2x2 对比。
run_one() {
  local name="$1" model="$2"
  echo
  echo "##########################################################"
  echo "# $name (训练格式 prompt)  <-  $model"
  echo "##########################################################"
  bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1
  sleep 5
  MODEL="$model" SERVED_NAME="$name" bash "$SETUP/06_serve.sh" start || { echo "❌ $name 起服务失败"; return 1; }
  ( cd "$VOPD_ROOT/eval" && "$VOPD_PY" infer.py \
      --benchmark vstar \
      --benchmark_json "$BJSON" \
      --out_dir model_answer \
      --model_name "${name}-trainfmt_seed42" \
      --seed 42 \
      --api_base "http://localhost:$PORT/v1/" \
      --model_id "$name" \
      --max_tokens 4096 \
      --parallel_workers 32 )
  local rc=$?
  bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1
  sleep 5
  return $rc
}

run_one "base-qwen35-4b" "Qwen/Qwen3.5-4B"
run_one "e1-step130" "$MERGE_ROOT/$TAG-step130"
run_one "e1-step195" "$MERGE_ROOT/$TAG-step195"
run_one "e1-step260" "$MERGE_ROOT/$TAG-step260"

# ---------- 3. 判分 + 2x2 汇总 ----------
# JSON 输出不能喂 first_letter_match(它会从 "bbox_2d" 之类的文本里抠错字母),
# 先按训练集同款规则抠 "answer" 字段, 抠不到再退回仓库规则。
echo
echo "=========================================================="
echo " 汇总: 训练格式 vs 裸 prompt (32 号作业)"
echo "=========================================================="
( cd "$VOPD_ROOT/eval" && "$VOPD_PY" - <<'PY'
import json, re, sys, importlib.util, collections
spec = importlib.util.spec_from_file_location("judge_qwenlm", "judge_qwenlm.py")
J = importlib.util.module_from_spec(spec); sys.modules["judge_qwenlm"] = J
spec.loader.exec_module(J)
_A = re.compile(r'"answer"\s*:\s*"?\(?([A-D])', re.I)
_BOX = re.compile(r'"bbox_2d"')

def grade(path, jsonfmt):
    items = [json.loads(l) for l in open(path) if l.strip()]
    ok = 0; bbox = 0; noans = 0
    per = collections.defaultdict(lambda: [0, 0])
    for it in items:
        gt = str(it["response"]).strip().upper()[:1]
        out = str(it["model_answer"])
        if _BOX.search(out): bbox += 1
        m = _A.search(out)
        if m:
            c = int(m.group(1).upper() == gt)
        else:
            if jsonfmt: noans += 1
            ex = J.extract_answer(out)
            c = int(bool(J.first_letter_match(gt, ex)))
        ok += c
        per[it["category"]][0] += c; per[it["category"]][1] += 1
    n = len(items)
    da, rp = per["direct_attributes"], per["relative_position"]
    return (100*ok/n, 100*bbox/n, noans, 100*da[0]/max(da[1],1), 100*rp[0]/max(rp[1],1), n)

print(f"{'模型':<18}{'prompt':>10}{'准确率':>8}{'带bbox':>8}{'无answer字段':>12}{'direct':>8}{'rel_pos':>8}")
for name in ["base-qwen35-4b", "e1-step130", "e1-step195", "e1-step260"]:
    bare = {"base-qwen35-4b": "base-qwen35-4b-rerun"}.get(name, name)
    try:
        a, b, no, da, rp, n = grade(f"model_answer/vstar/{bare}_seed42_answer.jsonl", False)
        print(f"{name:<18}{'裸':>10}{a:>7.2f}%{b:>7.1f}%{'—':>12}{da:>7.1f}%{rp:>7.1f}%")
    except FileNotFoundError:
        pass
    try:
        a, b, no, da, rp, n = grade(f"model_answer/vstar/{name}-trainfmt_seed42_answer.jsonl", True)
        print(f"{name:<18}{'训练格式':>10}{a:>7.2f}%{b:>7.1f}%{no:>12}{da:>7.1f}%{rp:>7.1f}%")
    except FileNotFoundError as e:
        print(f"{name:<18}{'训练格式':>10}  <缺文件: {e.filename}>")
PY
)
echo
echo " 结束 $(date)"
