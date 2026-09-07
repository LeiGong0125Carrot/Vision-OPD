#!/bin/bash
# 只做验证, 不装任何东西 —— 可以随便重跑。
# 注意: 不要为了验证去重跑 02b, 那会 pip 重装 vllm 从而覆盖掉已打的补丁。
set -uo pipefail          # 故意不加 -e: 验证脚本要把所有问题跑完再报, 不能中途退出
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
conda activate "$VOPD_ENV"

SP=$(python -c "import site; print(site.getsitepackages()[0])")
echo ">>> site-packages: $SP"
echo

echo "========== [1] vllm 的 .so: GLIBC 需求 =========="
for so in "$SP"/vllm/*.so "$SP"/vllm/vllm_flash_attn/*.so; do
  [ -f "$so" ] || continue
  m=$(readelf -V "$so" 2>/dev/null | grep -oE "GLIBC_[0-9]+\.[0-9]+" | sort -u -V | tail -1)
  [ -z "$m" ] && m="(无)"
  printf "  %-40s max=%s\n" "$(basename "$so")" "$m"
done
echo

echo "========== [2] 全环境扫描 glibc > 2.28 的残留 (约1-2分钟) =========="
BAD=0; N=0
while read -r so; do
  N=$((N+1))
  # `|| true` 是关键: 很多 .so 完全不含 GLIBC 版本串, grep 会返回 1
  m=$(readelf -V "$so" 2>/dev/null | grep -oE "GLIBC_[0-9]+\.[0-9]+" | sort -u -V | tail -1 || true)
  case "$m" in
    GLIBC_2.29|GLIBC_2.3[0-9]|GLIBC_2.[4-9][0-9])
      echo "    !! $m   $so"; BAD=$((BAD+1));;
  esac
done < <(find "$SP" \( -name "*.so" -o -name "*.so.*" \) 2>/dev/null)
echo "    扫描了 $N 个 .so"
if [ "$BAD" -eq 0 ]; then echo "    ✅ 全部 <= GLIBC_2.28"; else echo "    ❌ 发现 $BAD 个残留 (见上)"; fi
echo

echo "========== [3] 决定性测试: import vllm =========="
python - <<'PY'
import traceback, sys
def t(label, fn):
    try:
        fn(); print(f"  ✅ {label}")
    except Exception:
        print(f"  ❌ {label}"); traceback.print_exc(limit=3); sys.exit(1)

def imp_vllm():
    import vllm; print(f"     vllm {vllm.__version__}")
t("import vllm", imp_vllm)

def imp_moe():
    import vllm._moe_C          # 就是被打补丁的那个
t("import vllm._moe_C  (打过补丁的 .so)", imp_moe)

def imp_ops():
    import torch, vllm._custom_ops as ops
    assert hasattr(torch.ops, "_C")
t("vllm._custom_ops / torch.ops._C", imp_ops)

def imp_fi():
    import flashinfer; print(f"     flashinfer {getattr(flashinfer,'__version__','?')}")
t("import flashinfer (B200 的注意力后端)", imp_fi)

# 以下需要真 GPU。在 CPU 分区跑安装时自动跳过, 不算失败。
import torch
if not torch.cuda.is_available():
    print("  ⏭  无可见 GPU, 跳过 kernel 实跑测试 (在 CPU 节点上属正常)")
    sys.exit(0)

def moe_kernel():
    import torch
    from vllm.model_executor.layers.fused_moe import fused_topk
    h = torch.randn(4, 8, device="cuda", dtype=torch.float16)
    g = torch.randn(4, 4, device="cuda", dtype=torch.float16)
    out = fused_topk(h, g, 2, True)     # 0.18 返回 3 个值, 别按 2 个解包
    w, i = out[0], out[1]
    assert w.shape == (4, 2), w.shape
    torch.cuda.synchronize()            # 真正等 kernel 落地, 否则异步不报错
    print(f"     fused_topk -> {len(out)} 个返回值, weights{tuple(w.shape)} ids{tuple(i.shape)}")
t("在 B200 上实跑一个 MoE kernel", moe_kernel)
PY
rc=$?
echo
[ $rc -eq 0 ] && echo ">>> vllm 全部验证通过 ✅  下一步: nohup bash 03_build_ext.sh > build.log 2>&1 &" \
              || echo ">>> vllm 验证失败 ❌  需要转 apptainer 容器方案"
exit $rc
