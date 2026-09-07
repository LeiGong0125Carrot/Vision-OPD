#!/bin/bash
# 用论文同款 judge (openai/gpt-oss-120b, README:65) 统一复判全部 vstar 结果。
#
# 背景: 我们所有 e1/mb checkpoint 的数字都是 NO_JUDGE 纯规则严格下界; 8/22 的四个
# 旧数字虽带 judge 但是"模型自判"。论文 Qwen3.5-4B 基线 84.29 (=161/191) 用的是
# gpt-oss-120b judge —— 本作业把 model_answer/vstar/ 下全部 17 份结果文件在
# 同一个 judge 下重判, 得到与论文可比的整表。
#
# 判卷不重跑推理: judge_qwenlm.py 先规则后 LLM, 每份文件只有 ~20-70 条进 LLM。
# 权重已在 HF 缓存 (183G, 含 MXFP4); 96G 的 rtx_pro_6000 单卡装得下 (~63G 权重)。
#
# ⚠️ 已知坑: cal_acc.py 的 is_correct 要求 judge 字段 strip().lower() **恰好等于**
# "yes" —— 若 gpt-oss 回 "Yes." 会被判错。末尾的汇总脚本同时按"精确 yes"和
# "yes 前缀"两种口径统计并打印差异, 出现偏差时以后者为准并回来修。
#
#SBATCH --job-name=judge-gptoss
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-rtxpro6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/judge-%j.log
#SBATCH --open-mode=append

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"

PORT=8813
JUDGE_MODEL=openai/gpt-oss-120b
VLLM_BIN="$(dirname "$VOPD_PY")/vllm"
EVAL_DIR="$VOPD_ROOT/eval"

echo "=========================================================="
echo " gpt-oss-120b 统一复判  $(date)  节点=$(hostname)"
echo " GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "=========================================================="

# ---------- 0. 备份旧 judge 结果 (8/22 自判的四个, 会被覆盖) ----------
BK="$EVAL_DIR/judge/vstar_selfjudge_backup_20260827"
if [ -d "$EVAL_DIR/judge/vstar" ] && [ ! -d "$BK" ]; then
  cp -r "$EVAL_DIR/judge/vstar" "$BK"
  echo ">>> 旧 judge 结果已备份到 $BK"
fi

# ---------- 1. 起 judge 服务 (纯文本模型, 不走 06_serve.sh 的 VLM 模板) ----------
"$VLLM_BIN" serve "$JUDGE_MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192 \
  > "$SETUP/logs/gptoss-serve-$SLURM_JOB_ID.log" 2>&1 &
SRV=$!

ok=0
for i in $(seq 1 150); do   # 最多等 25 分钟 (权重 63G, weka 上加载几分钟)
  sleep 10
  if ! kill -0 $SRV 2>/dev/null; then
    echo "❌ vllm serve 进程退出了, 看 logs/gptoss-serve-$SLURM_JOB_ID.log 末尾:"
    tail -30 "$SETUP/logs/gptoss-serve-$SLURM_JOB_ID.log"
    exit 1
  fi
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then ok=1; break; fi
done
[ $ok -eq 1 ] || { echo "❌ judge 服务 25 分钟没就绪"; kill $SRV; exit 1; }
echo "✅ judge 服务就绪 (等了 $((i*10))s)"

# 冒烟: 一条最简 judge 调用, 确认回复形态
"$VOPD_PY" - <<PY
from openai import OpenAI
c = OpenAI(api_key="EMPTY", base_url="http://localhost:$PORT/v1/")
r = c.chat.completions.create(model="$JUDGE_MODEL",
    messages=[{"role":"user","content":"Your task is to judge whether the response expresses the same meaning as the answer of a question.\nThe question is: What color is the sky?\nThe answer is: (A) blue\nThe response is: A\nPlease check and compare them and then judge. If the response is correct, your output should be Yes. Otherwise, your output should be No. Directly give me your output."}],
    temperature=0, max_tokens=2048)
print("冒烟回复 content:", repr(r.choices[0].message.content))
PY

# ---------- 2. 逐文件复判 + 官方 cal_acc ----------
cd "$EVAL_DIR"
for f in model_answer/vstar/*_answer.jsonl; do
  tag=$(basename "$f" _answer.jsonl)
  echo
  echo "########## judge: $tag ##########"
  "$VOPD_PY" judge_qwenlm.py \
    --benchmark vstar \
    --model "$tag" \
    --api_base "http://localhost:$PORT/v1/" \
    --judge_model "$JUDGE_MODEL"
  "$VOPD_PY" cal_acc.py --benchmark vstar \
    --judge_json "judge/vstar/${tag}_answer.jsonl" \
    --benchmark_json vstar.json
done

kill $SRV 2>/dev/null

# ---------- 3. 双口径汇总表 ----------
echo
echo "=========================================================="
echo " 汇总 (judge=gpt-oss-120b; 精确yes = cal_acc 官方口径)"
echo "=========================================================="
"$VOPD_PY" - <<'PY'
import glob, json, os, re, collections
rows = []
for p in sorted(glob.glob("judge/vstar/*_answer.jsonl")):
    tag = os.path.basename(p)[:-len("_answer.jsonl")]
    data = json.load(open(p))
    n = len(data)
    exact = strictpfx = llm_used = 0
    weird = collections.Counter()
    per = collections.defaultdict(lambda: [0, 0])
    for it in data:
        j = str(it.get("judge", "")).strip()
        e = j.lower() == "yes"
        pfx = bool(re.match(r"^yes\b", j, re.I))
        if pfx and not e: weird[j[:40]] += 1
        if it.get("judge_source") == "llm": llm_used += 1
        exact += e; strictpfx += pfx
        per[it.get("category", "?")][0] += pfx
        per[it.get("category", "?")][1] += 1
    da, rp = per.get("direct_attributes", [0, 1]), per.get("relative_position", [0, 1])
    rows.append((tag, 100*exact/n, 100*strictpfx/n, llm_used,
                 100*da[0]/da[1], 100*rp[0]/rp[1], weird))
print(f"{'模型':<38}{'精确yes':>8}{'yes前缀':>8}{'LLM条数':>7}{'direct':>8}{'rel_pos':>8}")
for tag, e, s, l, da, rp, w in rows:
    flag = "  ⚠️两口径不一致" if abs(e - s) > 0.01 else ""
    print(f"{tag:<38}{e:>7.2f}%{s:>7.2f}%{l:>7}{da:>7.1f}%{rp:>7.1f}%{flag}")
    for k, v in w.most_common(3):
        print(f"      非标准yes样例 x{v}: {k!r}")
PY
echo
echo " 结束 $(date)"
