#!/bin/bash
# 多框输出格式(FMT_MB)的 base 合规性探针。
#
# 动机: 25_prep_data.py 里的单框 FMT 是经探针 22 实测筛选的, 新加的 FMT_MB 没有。
# 训练 job 18928899 已在 rtxpro6000 排队, 本探针在 B200 上并行跑 —— 若发现
# base 对多框格式合规率差(相对单框基线 93.9% 明显掉), 撤掉训练重新设计格式,
# 不损失队列位置以外的任何东西。
#
# 方法: 200 条训练题(seed=0 抽样) × 4 采样(temperature=1.0, 模拟训练 rollout),
# 对照单框探针的已知基线: S_json 格式合法 93.9% / IoU 0.368 / 准确率 38.8%。
#
#SBATCH --job-name=probe-mb
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/probe-mb-%j.log
#SBATCH --open-mode=append

set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"

export PORT=8812
echo "=== 多框格式探针  $(date)  节点=$(hostname) ==="
bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1; sleep 3
MODEL="Qwen/Qwen3.5-4B" SERVED_NAME="base-mb-probe" bash "$SETUP/06_serve.sh" start || exit 1

"$VOPD_PY" - <<'PY'
import json, re, base64, random, statistics as st, collections
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import pandas as pd

df = pd.read_parquet("/sfs/weka/scratch/nkw3mr/Vision-OPD/data/train_xy_rb_mb.parquet")
rnd = random.Random(0)
rows = [df.iloc[i] for i in rnd.sample(range(len(df)), 200)]
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8812/v1/", timeout=3600)

def uri(p):
    return "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()

def ask(r):
    msg = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": uri(r["images"][0]["path"])}},
        {"type": "text", "text": r["prompt"][0]["content"].replace("<image>", "").strip()},
    ]}]
    try:
        resp = client.chat.completions.create(model="base-mb-probe", messages=msg,
                                              max_tokens=1024, temperature=1.0, n=4)
        return [c.message.content or "" for c in resp.choices]
    except Exception as e:
        return [f"[ERR] {e}"] * 4

with ThreadPoolExecutor(16) as ex:
    outs = list(ex.map(ask, rows))

def iou(a, g):
    if not a or not g: return 0.0
    x1, y1 = max(a[0], g[0]), max(a[1], g[1]); x2, y2 = min(a[2], g[2]), min(a[3], g[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, g[2]-g[0])*max(0, g[3]-g[1]) - inter
    return inter/ua if ua > 0 else 0.0

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)
_ANS = re.compile(r'"answer"\s*:\s*"?\(?([A-D])', re.I)
_BOX = re.compile(r'"bbox_2d"\s*:\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]')

DUMP = open("/sfs/weka/scratch/nkw3mr/Vision-OPD-setup/probe_mb_outputs.jsonl", "w")
strict = 0; lenient = 0; n_tot = 0; errs = 0
why = collections.Counter()   # 严格合规失败的分类
nbox = collections.Counter(); mious = []; fious = []; accs = []; lens = []; bad_ex = []
for r, four in zip(rows, outs):
    g = list(r["extra_info"]["bbox_norm1000"]); al = str(r["extra_info"]["answer"]).strip().upper()[:1]
    for o in four:
        if o.startswith("[ERR]"): errs += 1; continue
        n_tot += 1; t = o.strip(); lens.append(len(t))
        DUMP.write(json.dumps({"gt_bbox": [int(v) for v in g], "answer": al, "output": o},
                              ensure_ascii=False) + "\n")
        m = _FENCE.search(t)
        if m: t = m.group(1)
        ok = False
        try:
            d = json.loads(t)
            if not isinstance(d, dict): why["JSON但不是dict"] += 1
            else:
                objs = d.get("objects")
                if not isinstance(objs, list) or not objs: why["缺objects列表"] += 1
                elif not all(isinstance(x, dict) and isinstance(x.get("bbox_2d"), list)
                             and len(x["bbox_2d"]) == 4 for x in objs): why["bbox_2d残缺"] += 1
                elif not all("label" in x for x in objs): why["缺label"] += 1
                elif not str(d.get("answer", "")).strip(): why["缺answer"] += 1
                else: ok = True
        except Exception:
            why["JSON解析失败"] += 1
        if ok: strict += 1
        boxes = [[float(v) for v in mm.groups()] for mm in _BOX.finditer(t)]
        if boxes: lenient += 1
        else: bad_ex.append(o[:110])
        nbox[min(len(boxes), 4)] += 1
        if boxes:
            ivals = [iou(b, g) for b in boxes]
            mious.append(max(ivals)); fious.append(ivals[0])
        else:
            mious.append(0.0); fious.append(0.0)
        ma = _ANS.search(t)
        accs.append(1 if (ma and ma.group(1).upper() == al) else 0)

DUMP.close()
print(f"\n===== 多框格式探针  200 题 × 4 采样 (temperature=1.0)  有效 {n_tot}, 请求失败 {errs} =====")
print(f"严格 schema 合规 (objects 列表+label+answer 完整): {100*strict/n_tot:.1f}%")
print("严格失败分类: " + "  ".join(f"{k}={100*v/n_tot:.1f}%" for k, v in why.most_common()))
print(f"宽松可解析 (至少抠出一个 bbox):                  {100*lenient/n_tot:.1f}%   [单框基线 93.9%]")
print(f"框数分布: " + "  ".join(f"{k}{'+' if k==4 else ''}框={100*v/n_tot:.1f}%" for k, v in sorted(nbox.items())))
print(f"IoU(max over boxes) 中位: {st.median(mious):.3f}   [单框基线 0.368, teacher 探针 0.803]")
print(f"IoU(第一个框)      中位: {st.median(fious):.3f}   (28_analyze_rollouts 将看到的口径)")
print(f"答案字母准确率:          {100*sum(accs)/len(accs):.1f}%   [单框基线 38.8%, ±2 为噪声]")
print(f"输出长度中位: {int(st.median(lens))} 字符")
if bad_ex:
    print("\n解析失败样例:")
    for x in bad_ex[:5]: print("  " + repr(x))
PY
rc=$?
bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1
echo "=== 结束 $(date) exit=$rc ==="
exit $rc
