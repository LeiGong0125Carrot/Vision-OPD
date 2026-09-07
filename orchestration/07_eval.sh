#!/bin/bash
# 跑评测。用法: bash 07_eval.sh [benchmark]   默认 vstar
#
# judge 的说明（重要）：
#   judge_qwenlm.py 是"先规则、后 LLM"：先用 mathruler + first_letter_match 判，
#   判不出来的才送 LLM。vstar 属于 MCQ_BENCHMARKS，模型输出 A/B/C/D，
#   绝大部分会被 first_letter_match 直接判掉，真正进 LLM 的通常只有少数几条。
#   但脚本有个前置硬检查：没配 judge 就 sys.exit(1)，哪怕一条都用不上。
#
#   所以这里默认把 judge 指向"已经起着的那个 server"，即模型给自己当裁判。
#   ⚠️ 这**只适合打通流程 / 自查**。论文用的是 openai/gpt-oss-120b，
#      要出可与论文对比的数字，必须换成独立的强判卷模型：
#        JUDGE_API_BASE=<外部API> JUDGE_MODEL=<模型名> bash 07_eval.sh vstar
#   脚本最后会打印各 judge_source 的条数，让你知道自判影响了多少条。
set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCH="${1:-vstar}"
PORT="${PORT:-8000}"
export API_BASE="${API_BASE:-http://localhost:$PORT/v1/}"
export OPENAI_MODEL_ID="${OPENAI_MODEL_ID:-Vision-OPD-9B}"
# NO_JUDGE=1: 完全不用 LLM judge, 只用规则判分。
# V* 等 MCQ 类基准这样做是**严格下界**, 因为 judge_qwenlm.py:247-270 里 LLM 只在规则
# 判为"不对"时才被调用 —— 它只能把错翻成对, 不会把规则判对的翻成错。
# 已实测验证: Vision-OPD-9B 在 vstar 上, 无 judge 的 90.05% 与带 LLM judge 的结果
# **完全一致**(分类也一致), 那 19 条送 LLM 的全被判成 "No"。
NO_JUDGE="${NO_JUDGE:-0}"
# JUDGE_MODEL_PATH: 用**本地 vLLM** 起判卷模型(judge_qwenlm.py:160 judge_via_vllm)。
# 这是论文的做法 —— README:65 写的 `openai/gpt-oss-120b` 是 OpenAI 放出的**开源权重**
# 模型(HF 上 openai/ 这个组织), 不是付费 API, 本地可完整复现。
# 必须清掉 JUDGE_API_BASE: judge_qwenlm.py:274 是 `if args.api_base: 走API else: 走vLLM`,
# API 模式优先, 不清的话本地模式根本轮不到。
#
# ⚠️ GPU 编排: judge_via_vllm 写死 gpu_memory_utilization=0.9, 和被测模型的服务
# (0.85)挤不进同一张卡。但 infer.py 只调 HTTP API、不占 GPU, 所以这样分卡即可:
#     服务:  CUDA_VISIBLE_DEVICES=0 bash 06_serve.sh start
#     评测:  CUDA_VISIBLE_DEVICES=1 JUDGE_MODEL_PATH=... bash 07_eval.sh
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-}"
if [ -n "$JUDGE_MODEL_PATH" ]; then
  export JUDGE_MODEL_PATH
  export JUDGE_API_BASE=""; export JUDGE_MODEL=""
  NO_JUDGE=0
elif [ "$NO_JUDGE" = "1" ]; then
  export JUDGE_API_BASE=""; export JUDGE_MODEL=""
else
  export JUDGE_API_BASE="${JUDGE_API_BASE:-http://localhost:$PORT/v1/}"
  export JUDGE_MODEL="${JUDGE_MODEL:-Vision-OPD-9B}"
fi
export BENCHMARK="$BENCH"
export PARALLEL_WORKERS="${PARALLEL_WORKERS:-32}"   # 单卡, 256 会把 server 打爆
# run_eval.sh 默认 MAX_TOKENS=32768(输出上限), 会和 server 的 --max-model-len 抢总长度配额。
# 这些 benchmark 大多是选择题, 训练时 MAX_RESPONSE_LENGTH 也才 1024, 4096 绰绰有余。
export MAX_TOKENS="${MAX_TOKENS:-4096}"

if ! curl -sf "http://localhost:$PORT/v1/models" >/dev/null; then
  echo "❌ localhost:$PORT 上没有服务。先跑: bash 06_serve.sh start"; exit 1
fi

# 光确认"有服务"不够 —— 还得确认它提供的正是我们要测的那个模型。
# 名字对不上时 infer.py 不会中止, 而是把 "[API_ERROR] 404 ..." 原样写进 model_answer,
# 判分再从错误文本里抠出字母, 最后算出一个看着像模像样的假分数(37.17% 那次就是这么来的)。
# 所以在跑 191 次推理之前先拦住。
served=$(curl -s "http://localhost:$PORT/v1/models" | "$VOPD_PY" -c \
  "import json,sys; print(' '.join(m['id'] for m in json.load(sys.stdin)['data']))" 2>/dev/null)
case " $served " in
  *" $OPENAI_MODEL_ID "*) : ;;
  *) echo "❌ 端口 $PORT 上的服务没有模型 '\''$OPENAI_MODEL_ID'\''"
     echo "   它实际提供: ${served:-<解析失败>}"
     echo "   要么用 SERVED_NAME=$OPENAI_MODEL_ID 重启服务, 要么把 OPENAI_MODEL_ID 改成上面那个名字。"
     exit 1 ;;
esac

echo ">>> benchmark      : $BENCHMARK"
echo ">>> 被测模型        : $OPENAI_MODEL_ID @ $API_BASE  (服务校验 ✓)"
if [ -n "$JUDGE_MODEL_PATH" ]; then
  echo ">>> judge          : 本地 vLLM  $JUDGE_MODEL_PATH  (GPU: ${CUDA_VISIBLE_DEVICES:-全部})"
elif [ "$NO_JUDGE" = "1" ]; then
  echo ">>> judge          : 不用 (纯规则判分, 得到严格下界)"
else
  echo ">>> judge          : $JUDGE_MODEL @ $JUDGE_API_BASE"
  [ "$JUDGE_API_BASE" = "http://localhost:$PORT/v1/" ] && \
    echo "    ⚠️  judge 就是被测模型本身 —— 仅用于打通流程, 不可用于论文对比"
fi
echo ">>> 并发           : $PARALLEL_WORKERS"
echo

cd "$VOPD_ROOT"
bash eval/run_eval.sh
rc=$?

# NO_JUDGE 模式下 run_eval.sh 会在 [3/4] judge 步失败(judge_qwenlm.py:193 的硬检查
# 没配 judge 就 sys.exit(1))。但 [2/4] 推理已经把 model_answer/ 写好了, 所以这里接手,
# 用规则自己判分。run_eval.sh 是 set -e, 所以它不会走到 [4/4] cal_acc.py。
if [ "$NO_JUDGE" = "1" ]; then
  echo
  echo ">>> run_eval.sh 在 judge 步退出(预期行为), 改用规则判分"
  for b in ${BENCHMARK//,/ }; do
    "$VOPD_PY" "$SETUP_DIR/16_score_norule.py" --benchmark "$b" \
      --model "${OPENAI_MODEL_ID}_seed${SEED:-42}" --by-category
  done
  exit 0
fi

echo
echo ">>> judge_source 分布（看有多少条真的走了 LLM judge）"
JF="$VOPD_ROOT/eval/judge/$BENCH"
"$VOPD_PY" - "$JF" <<'PY'
import json, sys, pathlib, collections
d = pathlib.Path(sys.argv[1])
RULE = {"mathruler", "first letter", "pope_exact", "mmvp_option"}
for f in sorted(d.glob("*_answer.jsonl")) if d.is_dir() else []:
    txt = f.read_text(encoding="utf-8").strip()
    try:                                  # 文件后缀是 .jsonl, 实际是格式化的 JSON 数组
        recs = json.loads(txt)
        if isinstance(recs, dict):
            recs = [recs]
    except json.JSONDecodeError:          # 退回按 JSONL 解析
        recs = [json.loads(l) for l in txt.splitlines() if l.strip()]
    c = collections.Counter(r.get("judge_source", "llm") for r in recs)
    tot = sum(c.values())
    print(f"  {f.name}  (n={tot})")
    for k, v in c.most_common():
        tag = "规则" if k in RULE else "LLM判"
        print(f"     {k:16} {v:5}  ({100*v/tot:5.1f}%)  [{tag}]")
    nllm = sum(v for k, v in c.items() if k not in RULE)
    yes_rule = sum(1 for r in recs
                   if r.get("judge_source") in RULE
                   and str(r.get("judge", "")).strip().lower().startswith("yes"))
    yes_all = sum(1 for r in recs if str(r.get("judge", "")).strip().lower().startswith("yes"))
    print(f"     -> 规则判对 {yes_rule}, 总判对 {yes_all}; LLM judge 贡献了 {yes_all - yes_rule} 个'正确'")
    print(f"     -> 换更强 judge 最多影响 {100*nllm/tot:.1f} 个百分点")
PY
exit $rc
