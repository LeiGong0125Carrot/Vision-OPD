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

# --- 7. center_u: 词表级恒定偏好被滤掉, 位置尖峰保留 ---
full_mask = torch.ones(B, T)
tok = torch.arange(K).unsqueeze(0).unsqueeze(0).expand(B, T, K).contiguous()  # 每位置同词表
# 构造: 恒定通道 c_k (每 token 类型一个常数) + 位置尖峰 (b0,t0,k0)
c = torch.randn(K + 1) * 0.5
u_const = c.expand(B, T, K + 1).contiguous()
spike = torch.zeros(B, T, K + 1); spike[0, 3, 2] = -2.0
# 用 f = h - u 反推出能产生该 u 的 null 側 (仅取前 K 列; 尾桶由归一化决定, 近似)
log_h_t = _sa_add_tail_log_probs(h.float())
u_target = u_const + spike
qc, mc = reconstruct_aha_target(h, (log_h_t - u_target)[..., :K], beta=1.0,
                                floor_alpha=None, response_mask=full_mask,
                                center_u=True, token_ids=tok)
qnc, _ = reconstruct_aha_target(h, (log_h_t - u_target)[..., :K], beta=1.0,
                                floor_alpha=None, response_mask=full_mask)
# 中心化后均值指标应接近 0, 且明显小于未中心化的 |u|
assert abs(mc["aha/u_centered_mean"]) < 0.05, f"中心化后均值应≈0, got {mc['aha/u_centered_mean']}"
# 恒定通道被滤掉 → qc 应比 qnc 更接近 p⁺ (恒定通道在 qnc 里扭曲 q, 在 qc 里不再扭曲)
# 位置尖峰保留 → 在 (0,3,2) 处 qc 仍显著低于 p⁺
d_spike = (qc[0, 3, 2] - h.float()[0, 3, 2]).item()
assert d_spike < -0.5, f"位置尖峰应保留负压, got {d_spike}"
# 无尖峰位置: qc ≈ p⁺ (中心化清掉了唯一的恒定通道)
d_clean = (qc[1] - h.float()[1]).abs().max().item()
assert d_clean < 0.35, f"纯恒定通道位置中心化后应≈p⁺, got {d_clean}"
# requires_grad 卫生
assert not qc.requires_grad

print("ALL TESTS PASSED")
