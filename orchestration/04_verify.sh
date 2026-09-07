#!/bin/bash
# 逐个 import 关键包 + 起一次最小 vLLM, 确认环境真的能用。
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
conda activate "$VOPD_ENV"

python - <<'PY'
import importlib, sys
mods = ["torch","torchvision","transformers","vllm","flashinfer","ray","datasets","accelerate",
        "peft","tensordict","hydra","omegaconf","qwen_vl_utils","flash_attn","causal_conv1d",
        "fla","liger_kernel","mathruler","math_verify","openai","PIL","cv2","pyarrow","verl"]
bad=[]
for m in mods:
    try:
        mod=importlib.import_module(m)
        print(f"  OK   {m:16} {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  FAIL {m:16} {type(e).__name__}: {e}"); bad.append(m)
print("\n失败:", bad or "无")
PY

echo; echo ">>> verl 训练入口能否 import + 解析 vopd config"
python -c "from verl.trainer.main_ppo import main; print('verl.trainer.main_ppo OK')"

echo; echo ">>> 冒烟测试 vLLM (小模型, 单卡)"
echo "    CUDA_VISIBLE_DEVICES=0 python -c \"from vllm import LLM; LLM('Qwen/Qwen2.5-0.5B-Instruct', max_model_len=2048).generate(['hi'])\""
