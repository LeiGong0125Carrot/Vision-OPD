#!/bin/bash
# 多框变体 vopd-xy_rb_mb-e1 的 V* 评测: step 65/130 x (裸 prompt / mb 训练格式)。
#
# 为什么只测 65/130: rollout 多框口径分析(2026-08-27)显示 step 65 是本轮最健康点
# (严格 MB 合规 93% / answer 存在 96.4%), 130 已在下坡(75%), 195/260 格式崩了一半
# 以上 —— 测了只会复现 e1 的单调伤害曲线, 不出新信息。
#
# 和 32/33 号的差别:
#   - 模型名前缀 mb-*, 不然会覆盖 e1-step130 等既有结果文件;
#   - 训练格式指令逐字取自 train_xy_rb_mb.parquet (FMT_MB, objects 数组),
#     33 号生成的 vstar_trainfmt.json 是单框 FMT, 对 mb 模型不是"训练同款";
#   - 补一格 base x mb格式 (base-qwen35-4b-mbfmt), 2x2 才有对照;
#   - base 裸 prompt 不重跑, 直接用 32 号已有的 base-qwen35-4b-rerun 结果。
#
# 卡型 B200: 评测走 vllm 自带 FA + FlashInfer JIT, arch 无关(见 32 号头注)。
#
#SBATCH --job-name=vopd-eval-mb
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/evalmb-%j.log
#SBATCH --open-mode=append

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
mkdir -p "$SETUP/logs"

TAG="${TAG:-vopd-xy_rb_mb-e1}"
CKPT_ROOT="$VOPD_ROOT/checkpoints/$TAG"
MERGE_ROOT="$VOPD_ROOT/merged"
STEPS="${STEPS:-65 130}"
export PORT="${PORT:-8811}"
BJSON="$SETUP/vstar_trainfmt_mb.json"

echo "=========================================================="
echo " V* 多框评测   tag=$TAG   steps=$STEPS"
echo " 节点=$(hostname)  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo " 判分: NO_JUDGE / 纯规则, 严格下界"
echo " $(date)"
echo "=========================================================="

# ---------- 1. 合并 FSDP 分片 (写独立目录, 不碰 checkpoints/) ----------
for s in $STEPS; do
  SRC="$CKPT_ROOT/global_step_$s/actor"
  DST="$MERGE_ROOT/$TAG-step$s"
  if [ -f "$DST/config.json" ] && ls "$DST"/*.safetensors >/dev/null 2>&1; then
    echo ">>> step $s 已合并, 跳过 ($(du -sh "$DST" | cut -f1))"
    continue
  fi
  [ -d "$SRC" ] || { echo "❌ 缺 $SRC"; exit 1; }
  echo ">>> 合并 step $s: $SRC -> $DST"
  mkdir -p "$DST"
  ( cd "$VOPD_ROOT" && "$VOPD_PY" -m verl.model_merger merge \
      --backend fsdp --local_dir "$SRC" --target_dir "$DST" )
  rc=$?
  [ $rc -ne 0 ] && { echo "❌ step $s 合并失败 rc=$rc"; exit 1; }
  echo "    ✅ $(du -sh "$DST" | cut -f1)"
done

# ---------- 2. 生成 mb 训练格式版 vstar json ----------
# 指令逐字取自 mb parquet; 选项 (A) -> A.; 剥掉 V* 自带的直接作答指令(191/191 条都有)。
"$VOPD_PY" - "$BJSON" <<'PY'
import json, re, sys
import pandas as pd
df = pd.read_parquet("/sfs/weka/scratch/nkw3mr/Vision-OPD/data/train_xy_rb_mb.parquet")
c = df.iloc[0]["prompt"][0]["content"]
i = c.find("Respond with exactly one JSON object")
assert i > 0, "mb parquet 里找不到格式指令"
INSTR = c[i:]
assert '"objects"' in INSTR, "指令不是多框版? " + repr(INSTR[:80])
print("指令原文:", repr(INSTR))

src = json.load(open("/sfs/weka/scratch/nkw3mr/Vision-OPD/eval/vstar.json"))
OPT = re.compile(r"^\(([A-D])\)\s*(.*)$")
TAIL = "Answer with the option"
out = []
for r in src:
    lines = [l for l in r["query"].strip().splitlines() if not l.strip().startswith(TAIL)]
    q, opts = [], []
    for l in lines:
        m = OPT.match(l.strip())
        (opts if m else q).append(f"{m.group(1)}. {m.group(2)}" if m else l)
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

# ---------- 3. 起服务 + 评测 ----------
serve_start() {
  bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1
  sleep 5
  # CHAT_TEMPLATE 走 06_serve.sh 默认(预填空 <think></think>), 与训练一致。
  MODEL="$2" SERVED_NAME="$1" bash "$SETUP/06_serve.sh" start
}

run_bare() {   # 裸 V* prompt, 走 07_eval.sh
  local name="$1" model="$2"
  echo; echo "########## $name (裸 prompt) <- $model ##########"
  serve_start "$name" "$model" || { echo "❌ $name 起服务失败"; return 1; }
  OPENAI_MODEL_ID="$name" NO_JUDGE=1 PARALLEL_WORKERS=32 bash "$SETUP/07_eval.sh" vstar
  local rc=$?
  bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1; sleep 5
  return $rc
}

run_mbfmt() {  # mb 训练格式 prompt, 直接调 infer.py
  local name="$1" model="$2"
  echo; echo "########## $name (mb 训练格式) <- $model ##########"
  serve_start "$name" "$model" || { echo "❌ $name 起服务失败"; return 1; }
  ( cd "$VOPD_ROOT/eval" && "$VOPD_PY" infer.py \
      --benchmark vstar \
      --benchmark_json "$BJSON" \
      --out_dir model_answer \
      --model_name "${name}-mbfmt_seed42" \
      --seed 42 \
      --api_base "http://localhost:$PORT/v1/" \
      --model_id "$name" \
      --max_tokens 4096 \
      --parallel_workers 32 )
  local rc=$?
  bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1; sleep 5
  return $rc
}

# base 只补 mb 格式这一格(裸 prompt 用 32 号已有的 base-qwen35-4b-rerun)。
run_mbfmt "base-qwen35-4b" "Qwen/Qwen3.5-4B"

for s in $STEPS; do
  run_bare  "mb-step$s" "$MERGE_ROOT/$TAG-step$s"
  run_mbfmt "mb-step$s" "$MERGE_ROOT/$TAG-step$s"
done

# ---------- 4. 判分 + 2x2 汇总 ----------
echo
echo "=========================================================="
echo " 汇总 (纯规则, 严格下界; e1/base 参考数取自既有结果文件重算)"
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

ROWS = [
    ("base",       "base-qwen35-4b-rerun_seed42_answer.jsonl",   "base-qwen35-4b-mbfmt_seed42_answer.jsonl"),
    ("mb-step65",  "mb-step65_seed42_answer.jsonl",              "mb-step65-mbfmt_seed42_answer.jsonl"),
    ("mb-step130", "mb-step130_seed42_answer.jsonl",             "mb-step130-mbfmt_seed42_answer.jsonl"),
    ("e1-step130(参考)", "e1-step130_seed42_answer.jsonl",       "e1-step130-trainfmt_seed42_answer.jsonl"),
]
print(f"{'模型':<18}{'prompt':>10}{'准确率':>8}{'带bbox':>8}{'无answer字段':>12}{'direct':>8}{'rel_pos':>8}")
for name, bare, fmt in ROWS:
    for path, tag, jf in ((bare, "裸", False), (fmt, "训练格式", True)):
        try:
            a, b, no, da, rp, n = grade(f"model_answer/vstar/{path}", jf)
            print(f"{name:<18}{tag:>10}{a:>7.2f}%{b:>7.1f}%{(str(no) if jf else '—'):>12}{da:>7.1f}%{rp:>7.1f}%")
        except FileNotFoundError:
            print(f"{name:<18}{tag:>10}  <缺文件: {path}>")
PY
)
echo
echo " 结束 $(date)"
