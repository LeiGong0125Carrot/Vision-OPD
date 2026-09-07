# State-Adaptive OPD 训练实现说明(2026-09-01/02)

本文档描述 `TreeVGR/09_01_plan.md`(含其附录修订)在 verl 代码库中的具体实现:精确到公式、
文件、配置旋钮与验证结果。方法背景、动机与组件验证见 09_01_plan;本文只写"实现成了什么样"。

---

## 1. 实现范围一句话

在现有 vopd(特权 OPD)训练路径上新增 **state_adaptive 分支**:冻结教师做两次前向
(特权视图 H + 全图参照 F),按超额效应门控逐 token 分配 KL 预算,二分求解重建强度 β_t,
以重建目标 q_t 替换 p^H 做 JSD 蒸馏。**v1 无 sharpening**(附录 §C 判决),
门控/β/loss 全程零标签。

## 2. 三个分布与共享支撑

每个响应 token 位置 t,同一学生前缀 h_t 上:

| 分布 | 模型 | 视觉输入 | 前向来源 |
|---|---|---|---|
| p^S | 可训练学生 | 全图 | 学生前向(带梯度)|
| p^H | 冻结教师 | region 特权视图 | 教师批次前向(no-grad)|
| p^F | 同一冻结教师 | 全图(**学生自己的输入**)| 新增第三次前向(no-grad)|

**支撑集 = 学生 top-100 索引 + tail bucket**(`distillation_topk=100, distillation_add_tail=True`)。
教师两次前向都在学生的 topk 索引上 gather(`topk_indices=student_topk_indices`),三分布同支撑,
`_sa_add_tail_log_probs` 补 tail 后各自归一。与 OPD-Aha 原设置(student top-100+tail)一致。

**F 视图的工程要点**:p^F 的输入就是学生的 `model_inputs` 原样传入 + `module=teacher_model`,
**trainer 侧不需要任何额外批次准备**(无 teacher_F_input_ids 之类)。学生 model_inputs 不含
`response_start_idx` 键,自动走"响应在末尾"约定,与学生前向逐键同路径。

## 3. 损失计算(compute_state_adaptive_distillation_loss)

全部门控量在 `torch.no_grad()` 内计算,目标 q_t 为常量;梯度只经 log p^S 进入 JSD。

```
u_t      = log p^H − log p^F                       # 含 tail bucket 的残差
e_t      = relu( D_JS(p^H, p^F) − λ · D_JS(p^H, p^S.detach()) )
g_raw_t  = σ( (e_t − τ_e) / T_e ) · loss_mask     # 无 sharpening, c_t ≡ 1
g_t      = max( g_raw_t, ρ · g_{t−1} )            # re-entry 窗
ε_t      = ε_max · g_t                            # 逐 token KL 预算
β_t      = binary_search( KL(q(β) ‖ p^H) ≤ ε_t ), β ∈ [0, β_max], 8 次迭代
q_t      = softmax( log p^H + β_t · u_t )         # detached 重建目标
loss_t   = 广义JSD_α( p^S, q_t )                  # α=0.5, 复用原 vopd 的混合公式, q 顶替教师槽
```

细节:

- **re-entry 的向量化**:递归 `g_t = max(g_raw_t, ρ g_{t−1})` 等价于
  `g_t = exp( t·logρ + cummax_s( log g_raw_s − s·logρ ) )`,log 域一次 cummax 完成,
  与朴素逐位递归逐元素相等(单测验证);`ρ=0` 时恒等返回(即 V2 消融)。
- **二分的单调性**:KL(q(β)‖p^H) 随 β 单调不减,lo=0 恒满足预算,8 次迭代分辨率 β_max/256。
  ε_t=0 时 β_t 精确 =0、q≡p^H,逐点退化为标准 OPD。
- **IS 权重与既有路径同款**:`is_clip=2.0` 的采样 token 重要性比 + rollout_is 权重,
  两臂(SA/V0)处理一致,保证可比。
- **门的关门底噪**:σ((0−τ_e)/T_e)=σ(−2)≈0.12(plan 公式固有),初始目标离 p^H ≤~0.06 nats,
  实测无害(§7)。
- **上报指标**(tensorboard `state_adaptive/*`):e_t_mean, js_hf_mean, js_hs_mean,
  gate_mean, gate_raw_mean, gate_frac_open(>0.5 占比), beta_mean, eps_mean, kl_q_h_mean。

## 4. 代码改动清单

| 文件 | 改动 |
|---|---|
| `verl/trainer/ppo/core_algos.py` | 新增 `compute_state_adaptive_distillation_loss` 及辅助 `_sa_add_tail_log_probs` / `_sa_js_divergence` / `_sa_reentry_scan` |
| `verl/workers/actor/dp_actor.py` | self_distillation 分支内:`state_adaptive` 开关读取;教师 H 前向后新增 F 视图第三次前向(model_inputs + module=teacher_model + student topk 索引);loss 分支切换;计时键 `timing_s/update_actor/teacher_full_forward` |
| `verl/workers/config/actor.py` | `SelfDistillationConfig` 新增 8 个字段 + `__post_init__` 校验(state_adaptive 强制要求 full_logit + topk + add_tail)|
| `verl/trainer/config/actor/actor.yaml` | self_distillation 段新增同名默认值(Hydra struct 模式下必须先声明)|
| `scripts/prepare_rl37k_state_adaptive.py` | 训练数据准备(§5)|
| `scripts/run_state_adaptive_4b.sh` | 4B 双臂训练入口(§6)|
| `Vision-OPD-setup/50_train_sa4b.sbatch` | slurm 提交(§6)|

## 5. 训练数据(train_sa4k.parquet)

来源 `data/TreeVGR-RL-37K/subset4k_adv_v2.parquet`(任务加权 4000 条:spatial_rel 1700 /
color 1500 / state_pose 400 / material 300 / ocr 100;**counting 已整类剔除**——GT 框数≡答案,
任何框特权都在漏标签)。`prepare_rl37k_state_adaptive.py` 产出:

- **teacher 特权形态 = region**:每框一张 2.42× 外扩裁剪 + 红框(几何与
  `eval/treebench_probe/infer_privilege.py::region_crop` 同源;RL-37K 过户验证的冠军形态,
  逐任务类 ≈2× hide)。裁剪短边 < 28px 时对称扩到 28(Qwen patch 下限保险)。
  渲染到 `teacher_images_region/{orig_row}_{k}.jpg`。
- **多框对接**:一题 1–3 框 → 1–3 张 crop。teacher 侧用 `teacher_prompt` 模板列
  (n 个 `<image>` 占位 + 纯问题),走 `_build_teacher_messages_from_template` 路径,
  绕开 `_swap_images_in_messages` 的"图数须相等"校验。
- **文本两侧全裸**:学生 prompt = 原 problem(`<image>`+纯问题),teacher_prompt 文本 = 同一纯问题。
  **无任何特权句/标签句**(链条分析证实特权句在 4B 上制造 40% 的指令缠斗,为最大伤害通道)。
- ground_truth 仅存于 reward_model/extra_info 供事后分析,训练路径(reward-free vopd)不读取。
- 全 4000 条渲染成功;prompt 实测(含视觉 token)p99≈830、最大≈4016 → `MAX_PROMPT_LENGTH=4608`。

## 6. 训练设置与启动

- **冻结同尺度教师**:`teacher_model_source=legacy + teacher_regularization=ema +
  teacher_update_rate=0.0` → teacher ≡ ref module(初始权重),rate=0 时 EMA 更新直接跳过
  (dp_actor:154)。两次教师前向共享这一个冻结实例。**resume 语义安全**(教师每次启动从底模
  重载,无 e1 的 EMA 重置陷阱)。
- **入口**:`scripts/run_state_adaptive_4b.sh`。`SA_ENABLE=False` 即标准特权 OPD 基线(V0),
  两臂除方法开关外逐字节同配。主要默认:batch 96 / mini 96 / rollout n=8 / lr 2e-6 /
  α=0.5 / prompt 4608 / response 1024 / topk 100 / is_clip 2.0。
- **slurm**:`sbatch 50_train_sa4b.sbatch`(2×rtx_pro_6000 / 600G / save_freq=14);
  V0 臂 `sbatch -J v0opd4b --export=ALL,SA_ENABLE=False ...`;卡数随 `--gres` 自动适配。
- **旋钮与默认值**(均可 `--export=ALL,VAR=...` 覆写):

| 旋钮 | 默认 | plan §18 对照 |
|---|---|---|
| SA_LAMBDA (λ) | 1.0 | 一致 |
| SA_TAU_E (τ_e) | 0.03 | **偏离**:plan 建议 running quantile;自举期 e≡0 会把分位数 τ 定成 0、开门过早,故用固定值+监控(见 §8)|
| SA_TEMP_E (T_e) | 0.015 | 按探针 D_JS 量纲定 |
| SA_RHO (ρ) | 0.9 | 一致 |
| SA_EPS_MAX (ε_max) | 0.5 nats | plan 未给数;可调 |
| SA_BETA_MAX | 8.0 | 一致(6 or 8)|
| SA_BINARY_ITERS | 8 | 一致(6~8)|

- **消融开关映射**(plan §16):V0 = `SA_ENABLE=False`;V1(固定β)= `SA_TAU_E=-999`(门钉满,
  预算恒为 ε_max);V2(仅 e_t 门)= `SA_RHO=0`;V4(全量)= 默认。V5(mean-RGB 参照)未实现。

## 7. 验证记录

**CPU 单测**(合成分布):自举性质(S≡F ⇒ e_t=0,门=底噪 0.119,目标离 p^H ~0.06 nats);
逐 token KL(q‖p^H)≤ε_t 严格成立;re-entry 扫描 == 朴素递归;S≡H 且 H≠F 时门开(gate 0.85 /
β 4.9,plan §5.2 场景);mask 行/尾部 padding 零梯度;ε=0 时 β 精确 =0;bf16 有限。

**GPU 冒烟**(单卡,batch 8×n2):step 2 上 `js_hf ≡ js_hs` 至 7 位小数(自举恒等式在真模型
复现),e_t=0、gate=σ(−2) 精确命中;20 步后 e_t→0.030、门半开、KL≤ε 守恒。
单卡教训:vLLM util 需降(权重不分片挤占);动态 bsz 的 token 上限不得低于填充宽度;
单卡兜底 = `use_dynamic_bsz=False + ppo_micro_batch_size_per_gpu=1`。

**双臂整轮**(2 卡,job SA 19173897 / V0 19174021,各 41 步 = 1 epoch,~1.4h,exit 0):

- 机制曲线:e_t 自举 0 → 0.013 稳态;gate 0.36–0.44,frac_open ~30–35%(**未饱和,
  τ_e=0.03 无需调**);β 1.4 → 2.2;kl_q_h ≤ eps 全程成立;
- 每步账单(中位):token 289k;学生前向 13.5s + 教师 H 14.3s + **教师 F 16.8s** + 反传 30.8s;
  SA 整步 115s vs V0 108s —— **双教师代价 ≈ +7%**;
- 行为:响应长度 SA 184→85 词(V0 184→127),SA 压缩更强;
- checkpoint:step 14/28/41 ×2 臂,`checkpoints/{SA-OPD,StdOPD-region}-Qwen3.5-4B/`。

## 8. 与 plan 的偏离清单(均为有意,已论证)

1. **τ_e 固定而非分位数**:保护自举期(分位数在 e≡0 时会立即半开门)。监控点:
   `gate_frac_open` 若前 10 步冲到 ~1,改 `SA_TAU_E=0.06` 重跑。实测 41 步稳在 35%,未触发。
2. **rollout n=8 而非 §22 的 single rollout**:8 条独立蒸馏、无任何选择机制
   (不违反 no-BoN/no-K-sampling 红线),沿用 e1 惯例且两臂同配。
3. **关门底噪 σ(−2)≈0.12**:plan 门公式固有,非实现引入;冒烟实测无害。
4. **MIN_CROP=28 保险**:探针版 region_crop 无此保护,仅影响退化小框。
5. 支撑集用学生 topk(继承既有 vopd 实现,与 OPD-Aha 原文一致)——教师支撑外质量落入 tail bucket。

## 9. 待办与展望

- TreeBench + V* 评测(2 臂 × 3 ckpt;51_eval_sa4b.sbatch 已备,首投被系统 CANCELLED by 0,待重投);
- OPSA 基线 loss(证明"真的在蒸馏"的对照)未实现;
- V1 固定 β 消融待跑(隔离"参照系修改"与"自适应门控"两个贡献);
- 9B 主实验(TreeVGR 49.63 为目标线;两视图 oracle 并集:4B 57.28 / 9B 61.23,
  方法只需吃下选择头寸的一半即大幅超线);
- 训练后 Probe(plan §14):gate 与免标签失败标记(特权下残余搜索——无句设定下唯一存活的
  词面标记)的对齐;e_t 增长曲线作为漂移-兑现读数。
