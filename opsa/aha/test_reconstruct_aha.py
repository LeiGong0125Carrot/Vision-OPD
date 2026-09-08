#!/usr/bin/env python
"""reconstruct_aha_target CPU 单测 (纯 torch, 无 GPU).

用法: $VOPD_PY opsa/aha/test_reconstruct_aha.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from verl.trainer.ppo.core_algos import _sa_add_tail_log_probs, reconstruct_aha_target  # noqa: E402

torch.manual_seed(0)
B, T, K = 2, 5, 8


def rand_topk_logps():
    """模拟 '全词表 log_softmax 在 top-k 上 gather' 的输出: 概率和 < 1, 降序."""
    raw = torch.randn(B, T, K + 4)
    logp = torch.log_softmax(raw, dim=-1)
    return logp.sort(dim=-1, descending=True).values[..., :K]


h = rand_topk_logps()   # p⁺
f = rand_topk_logps()   # p⁰


def full_q(logq_k):
    """K 列 -> 补尾桶成 K+1 真分布 (与 compute_self_distillation_loss add_tail 同法)."""
    return _sa_add_tail_log_probs(logq_k)


# --- 1. β=0 恢复 p⁺ ---
q0, m0 = reconstruct_aha_target(h, f, beta=0.0, floor_alpha=None)
assert torch.allclose(q0, h.float(), atol=1e-5), "β=0 应精确恢复 p⁺ 的前 K 列"

# --- 2. 无 floor 时 log-odds 差 = β·u (K+1 全支持上) ---
beta = 4.0
qb, mb = reconstruct_aha_target(h, f, beta=beta, floor_alpha=None)
log_h = _sa_add_tail_log_probs(h.float())
log_f = _sa_add_tail_log_probs(f.float())
u = log_h - log_f
# 返回的 K 列上直接验证 (避免 float32 尾桶重建的条件数误差):
i, j = 0, K - 1
lhs = qb[..., i] - qb[..., j]
rhs = (log_h[..., i] - log_h[..., j]) + beta * (u[..., i] - u[..., j])
assert torch.allclose(lhs, rhs, atol=1e-4), "log-odds 差应等于 Δlog_h + β·Δu"
# 与手算 K+1 重建精确一致 (含尾桶维)
internal = torch.log_softmax(log_h + beta * u, dim=-1)
assert torch.allclose(qb, internal[..., :-1], atol=1e-6), "K 列应与手算重建逐元素一致"
# 关键契约: 损失侧 add_tail 从 K 列重建出的尾桶 == q 内部的真尾桶 (fp32)
assert torch.allclose(full_q(qb), internal, atol=1e-3), "add_tail 重建的 K+1 分布应还原内部 q (含尾桶)"
assert not qb.requires_grad, "重建 q 不得携带梯度"

# --- 3. 归一化守恒: 返回的 K 列概率和 < 1, 补尾后 K+1 和 = 1 ---
assert (qb.exp().sum(-1) < 1.0 + 1e-6).all(), "K 列概率和必须 < 1 (留给尾桶)"
assert torch.allclose(full_q(qb).exp().sum(-1), torch.ones(B, T), atol=1e-5)

# --- 4. floor 只动谷区负 u 维 ---
alpha = 0.1
qf, mf = reconstruct_aha_target(h, f, beta=beta, floor_alpha=alpha)
valley = log_h < math.log(alpha) + log_h.max(dim=-1, keepdim=True).values
clamp = valley & (u < 0)
u_gated = torch.where(clamp, torch.zeros_like(u), u)
expected = torch.log_softmax(log_h + beta * u_gated, dim=-1)[..., :-1]
assert torch.allclose(qf, expected, atol=1e-5), "floor 版 q 与手算 gated 重建不一致"
# 头部维 (非谷区) 在两版中 log-odds 关系不变: 取每个位置的前两维 (top-2, 几乎必属头部),
# 仅在两维都非谷区的位置断言 floor 前后 log-odds 差一致
both_head = ~valley[..., 0] & ~valley[..., 1]
assert both_head.any(), "构造数据下应存在 top-2 双头部位置"
lo_repro = (qb[..., 0] - qb[..., 1])[both_head]
lo_floor = (qf[..., 0] - qf[..., 1])[both_head]
assert torch.allclose(lo_repro, lo_floor, atol=1e-5), "floor 不得改变头部维之间的 log-odds"
# 谷区负 u 维: floor 版概率 >= 原版 (负压被解除)
assert (qf.exp()[clamp[..., :-1]] >= qb.exp()[clamp[..., :-1]] - 1e-6).all(), \
    "被 clamp 的谷区维概率不应低于原版"

# --- 5. metrics 合理性 ---
for mm, name in [(m0, "β=0"), (mb, "repro"), (mf, "floor")]:
    for k, v in mm.items():
        assert isinstance(v, float) and math.isfinite(v), f"{name} metric {k} 非有限 float: {v}"
assert mf["aha/floor_clamped_frac"] > 0, "构造数据下 floor 应有触发"
assert mb["aha/floor_clamped_frac"] == 0.0, "repro 臂 clamped_frac 恒 0"
assert mb["aha/u_neg_valley_frac"] > 0, "repro 臂应观测到谷区负压暴露"

# --- 6. response_mask 过滤不改变 q, 只改 metrics ---
mask = torch.zeros(B, T)
mask[:, :2] = 1.0
qm, _ = reconstruct_aha_target(h, f, beta=beta, floor_alpha=None, response_mask=mask)
assert torch.allclose(qm, qb), "response_mask 只影响 metrics, 不得影响 q"

print("ALL TESTS PASSED")
