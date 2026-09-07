#!/bin/bash
# vLLM 冒烟测试: 用一个小模型真跑一次推理。
# 这是第一次让 vLLM 在 B200 上实际执行 kernel —— 前面所有验证都只到 import 为止。
#
# 重点观察三件事:
#   1) vLLM 选了哪个 attention backend (B200 上应该是 FlashInfer, 不该是 FA3)
#   2) FlashInfer 首次 JIT 编译 sm_100 kernel 的耗时 (可能几分钟, 不是卡死)
#   3) 生成结果是否正常
set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
conda activate "$VOPD_ENV"

SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

# --- keepalive 会占显存 + 抢时间片, 跑真实负载前必须停 ---
if [ -f "$SETUP/.keepalive.pid" ] && kill -0 "$(cat "$SETUP/.keepalive.pid")" 2>/dev/null; then
  echo ">>> 检测到 keepalive 在跑, 先停掉 (它的 CUDA context 会占显存)"
  bash "$SETUP/keepalive.sh" stop
  sleep 3
fi

echo ">>> GPU 初始状态"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo
echo ">>> 模型: $MODEL"
echo ">>> 注意: FlashInfer 要为 sm_100 现场 JIT 编译 kernel, 首次启动慢几分钟是正常的"
echo

python - "$MODEL" <<'PY'
import sys, time, os
model = sys.argv[1]

t0 = time.time()
from vllm import LLM, SamplingParams
print(f"[{time.time()-t0:6.1f}s] import vllm 完成", flush=True)

t1 = time.time()
llm = LLM(model=model, max_model_len=2048, gpu_memory_utilization=0.35,
          enforce_eager=True, trust_remote_code=True)
print(f"[{time.time()-t1:6.1f}s] 引擎初始化完成 (含 FlashInfer JIT)", flush=True)

t2 = time.time()
out = llm.generate(
    ["请用一句话解释什么是强化学习。", "The capital of France is"],
    SamplingParams(temperature=0.0, max_tokens=48),
)
print(f"[{time.time()-t2:6.1f}s] 生成完成", flush=True)

print("\n--- 生成结果 ---")
for o in out:
    print(f"  prompt : {o.prompt!r}")
    print(f"  output : {o.outputs[0].text.strip()!r}")
    print(f"  tokens : {len(o.outputs[0].token_ids)}")
print(f"\n总耗时 {time.time()-t0:.1f}s")
PY
rc=$?

echo
echo ">>> GPU 结束状态"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo
if [ $rc -eq 0 ]; then
  echo "✅ vLLM 在 B200 上跑通了。下一步: 起 Vision-OPD-9B 的 server"
  echo "   vllm serve yuanqianhao/Vision-OPD-9B --tensor-parallel-size 1 \\"
  echo "       --gpu-memory-utilization 0.85 --served-model-name Vision-OPD-9B --trust-remote-code"
else
  echo "❌ 失败 (exit $rc), 把上面的报错贴出来"
fi
exit $rc
