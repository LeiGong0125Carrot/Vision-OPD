# Floor 臂算法档案（plausibility floor on the Aha reconstruction）

本文件是 floor 臂的完整参考：算法定义、与原版 OPD-Aha 的精确差异、实现细节与坑、训练配置、
动力学读数、全量评测结果（**judge 口径**）、与后续臂的机制关系、以及归因缺口。
写作日期 2026-09-23。代码 commit：`ef20ab51`（引入）→ `d92604d`（review 修复）。

**为什么单独立档**：`Aha-floor-Qwen3.5-4B-6karmA`（下称 ahafloor6k）是本季 **TreeBench judge
读数最高的臂之一（峰 51.60）**，高于我们自训 A 基线（`pair_ahaA_6karmA`，49.38）。它的核心机制
（不压低概率 token）后来成为 supn8/ST 两个臂的思想来源，因此需要一份可引用的精确记录。
**注意：它的 TB 优势存在归因缺口，见第八节——不可直接当作"floor 算法优于 A"的证据。**

---

## 一、算法定义

记 p⁺ = 特权视图 teacher 分布，p⁰ = 空白视图 teacher 分布，二者都在**学生 top-k 支持**上
取值（K 维，全词表归一化后 gather，再各自补尾桶成 K+1 维真分布）。

视觉增量：

$$u(v) = \log p^+(v) - \log p^0(v)$$

原版 OPD-Aha（revision_opd Eq.12）的重建目标：

$$\log q = \mathrm{log\_softmax}\big(\log p^+ + \beta \cdot u\big)$$

**floor 只改一处**——先对 u 施加一道门再重建：

$$
u_{\text{gated}}(v) =
\begin{cases}
0, & \text{if } p^+(v) < \alpha \cdot \max_w p^+(w) \ \wedge\ u(v) < 0 \\[4pt]
u(v), & \text{otherwise}
\end{cases}
$$

$$\log q = \mathrm{log\_softmax}\big(\log p^+ + \beta \cdot u_{\text{gated}}\big)$$

即：**谷区（相对低概率区）的负向视觉增量被截为 0 —— 负力只落在头部。**
$\alpha$ = `floor_alpha` = 0.1（谷区判据：p⁺ 低于峰值的 10%）。$\alpha$ = None 即退化为原版。

**除此之外与原版逐字节相同**：anchor 仍是 p⁺、仍走全支持 softmax 归一化、support 仍是学生
top-100、尾桶补法相同、下游仍是 α=0.5 的 JSD（Eq.13）、正向 u 全域完整保留（**含谷区正向**）。

---

## 二、代码

实现：`verl/trainer/ppo/core_algos.py::reconstruct_aha_target`（`:1214`），floor 分支在 `:1299-1303`：

```python
with torch.no_grad():
    log_h = _sa_add_tail_log_probs(teacher_topk_logps.float())        # (B,T,K+1)  log p⁺
    log_f = _sa_add_tail_log_probs(teacher_null_topk_logps.float())   # (B,T,K+1)  log p⁰
    u_pre = log_h - log_f
    u_raw = u_pre                       # center_u 可选,施加顺序 center → floor → 重建
    ...
    elif floor_alpha is not None:
        valley     = log_h < math.log(floor_alpha) + log_h.max(dim=-1, keepdim=True).values
        clamp_mask = valley & (u_raw < 0)
        u          = torch.where(clamp_mask, torch.zeros_like(u_raw), u_raw)
        log_q      = torch.log_softmax(log_h + beta * u, dim=-1)
    else:                               # 原版 A
        log_q      = torch.log_softmax(log_h + beta * u, dim=-1)
return log_q[..., :-1], metrics         # 只返回前 K 列
```

接入方式：q 替换 `teacher_topk_logps` 后，走与 V0 **完全相同**的
`compute_self_distillation_loss`（α=0.5），对照最干净。第三个 `no_grad` teacher 前向吃 null
输入，计时键 `teacher_null_forward`。

**三个不能省的实现细节**

1. **返回 K 列，切勿传 K+1**。函数内部补尾桶只为让 u 和 softmax 在真分布上计算；返回时丢掉
   尾桶列，由损失侧 `distillation_add_tail=True` 精确还原。传 K+1 会导致尾桶被算两次。
2. **尾桶天然属谷区，同规则**。尾桶的 p⁺ 通常远小于 0.1·max，因此尾桶的负 u 也被截 0 ——
   这是设计意图（不压尾巴），不是疏漏。
3. **相对阈值，非绝对**。`log(α) + max(log_h)` 是逐位置的相对线，随该位置的分布尖锐度自适应；
   绝对阈值会在平坦位置把几乎全部 token 划进谷区。

**哨兵指标**（`aha/*`，两臂都记以保可比）

| 键 | 含义 | floor 臂实测 |
|---|---|---|
| `aha/u_mean` | **pre-center pre-clamp** 的 u 均值（恒用原始 u，全臂可比） | −0.685 → −0.458 |
| `aha/u_neg_valley_frac` | 谷区∧负向 的维占比 | 0.632 → 0.654 |
| `aha/floor_clamped_frac` | 实际被截断的维占比（α=0.1 时定义与上式重合 = 交叉验证） | 0.632 → 0.602 |
| `aha/q_tail_mass` | 重建后尾桶质量 | ~0.0018 |

---

## 三、理论动机，以及我们的实测反驳

**动机**：Ren ICLR25 的**挤压效应**（squeezing）——负梯度施加在低概率 token 上，会把概率质量
挤向已有头部，引发熵塌缩与尾部瘦身。floor 通过"负力只落头部"避免触发。

**我们的实测反驳（重要）**：repro（原版）与 floor 两臂各跑 41 步，**两臂均零挤压**——
`argmax_conf` 0.82→0.63-0.67（ahafloor6k 上 0.949→0.899）**反而下降**，响应长度稳定
（257→236），loss 稳 ~0.23。同期 OPSA 的 advantage 形式在同域 16-22 步就崩。

结论：**重建式（JSD 质量加权）的负压天然免疫挤压，只有 advantage 形式才触发。**
也就是说 floor 想防的病，在这套重建框架里本来就不发作 —— **floor 的理论依据在我们的体制下
不成立**。它若有经验增益，机制必须另找解释（候选：不压谷区 = 保留替代续写入口的采样概率，
与"防挤压"是不同的故事）。

**一个我们自己发现的修正**（写在 zone 臂注释 `:1272`）：Ren 挤压的变量是**学生分布 p_S，
不是 p⁺**。floor 用 p⁺ 判头部只是近似；理论上正确的门应该是学生头部。后续 ST 臂的
`p⁺ 头部 ∩ 学生头部` 双条件正是这个修正的落地。

---

## 四、训练配置（ahafloor6k，即 TB 最高读数那一次）

| 项 | 值 |
|---|---|
| 实验名 | `Aha-floor-Qwen3.5-4B-6karmA` |
| 数据 | `data/TreeVGR-RL-37K/train_6karmA_aha.parquet`（2459 行） |
| **视图（关键）** | **非 pair**：teacher 正 = **crop 单图**，null = **均值块单图**（`bbox_images`/`null_images` 各 1 元素） |
| β | 4.0 |
| floor_alpha | 0.1 |
| batch / rollout_n | 48 / **2** |
| epochs / 总步数 | **2 epoch → 102 步**（51 步/epoch） |
| save_freq | 5（22 个 ckpt：5..51, 55..100, 102） |
| 模型 | Qwen3.5-4B |
| 硬件 | 2× RTXPro6000，~2.5 min/步，driver 内存 ~330G/512G |
| launcher | `AHA_FLOOR_ALPHA=0.1 TASK_TRAIN_FILE=.../train_6karmA_aha.parquet EXPERIMENT_SUFFIX=6karmA bash opsa/aha/run_aha_4b.sh` |
| 训练日志 | `opsa/logs/train_floor_6karmA_ep2.log` |

**视图差异务必注意**（这是后续归因的关键）：

| | `6karmA_aha`（floor 用） | `6karmA_pair`（A/supn8/ST 用） |
|---|---|---|
| 学生看 | 全图 | 全图 |
| p⁺ | crop 单图 | [全图, crop] 双图 |
| p⁰ | 均值块单图 | [全图, 均值块] 双图 |
| u 的语义 | crop 相对空白的**全部**信息差（混杂"有无上下文"） | 共享全图上下文下 crop 的**增量**贡献 |

pair 视图的设计依据是 qcontrast 探针得到的**结构性定律**：负视图必须与正视图共享前缀承诺
结构，否则对比被承诺失配主导。非 pair 视图的 u 绝对值更大（`u_mean` −0.69 vs pair 体制的量级）
但更脏。

---

## 五、全量评测结果（judge 口径，gpt-oss-120b；**规则分不入表**）

TreeBench（n=405）与 V\*（n=191），22 个 checkpoint：

| step | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 51 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TB | 48.40 | 46.67 | 48.64 | 47.16 | 49.88 | 49.88 | 48.15 | **38.77** | 50.86 | **51.60** | 50.12 |
| V\* | 84.82 | 85.34 | 87.96 | 87.43 | 87.43 | 89.53 | 86.91 | **71.20** | 88.48 | 89.53 | 88.48 |

| step | 55 | 60 | 65 | 70 | 75 | 80 | 85 | 90 | 95 | 100 | 102 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TB | 46.67 | 48.64 | 49.88 | 49.38 | 48.89 | 49.14 | 49.88 | 48.64 | 46.67 | 49.63 | 50.12 |
| V\* | 91.10 | 89.01 | 89.01 | 89.01 | 91.10 | **91.62** | 90.05 | 89.53 | 90.05 | 89.53 | 90.58 |

- **TB 峰 51.60 @50**；后半段（55-102）均值 ~48.9
- **V\* 峰 91.62 @80**；后半段均值 ~90.1
- **step40 双基准同时崩**（TB 38.77 / V\* 71.20）：两个独立基准同时掉 10+ 点，说明是该
  checkpoint 自身异常（格式漂移或存档问题），不是评测噪声 —— 使用该臂时应跳过 step40。
- **相邻步摆动大**：TB step50=51.60 而 step55=46.67（5 点），是"TB 单点不可信"的典型证据。

### 对照（全部 judge 口径）

| 臂 | 视图 | 算法 | 步数 | V\* 峰 | TB 峰 |
|---|---|---|---|---|---|
| base Qwen3.5-4B | — | — | — | 83.25 | 44.94 |
| **ahafloor6k** | 非 pair | **floor** | 102 | 91.62 | **51.60** |
| `aha-center-floor-...-6karmA` | 非 pair | center+floor | 51 | — | 50.37 |
| `aha-center-...-6karmA` | 非 pair | center | 51 | — | 49.14 |
| `pairfloor6k` | pair | floor | 51 | **未测** | 51.36 |
| `pair_ahaA_6karmA`（我们自训 A） | pair | 原版 | 51 | 91.62 | 49.38 |
| `ahapair6k`（官方 ckpt） | pair | 原版 | 51 | **93.72** | 50.37 |
| `supn8`（本周） | pair | u⁻ tilt | 51 (n8) | 91.62 | 48.64 |

---

## 六、机制关系：四象限对照（floor / supn8 / ST 的统一视图）

把 support 按 (p⁺ 头部 / 谷区) × (u 正 / 负) 切四格，看每臂施加的力：

| 象限 | A（原版） | **floor** | **supn8** | **ST (stn8)** |
|---|---|---|---|---|
| 头部 · u>0 | β·u | β·u | **0** | 等比接收 |
| 头部 · u<0 | β·u | β·u | β·u | **守恒扣减 δ** |
| 谷区 · u>0 | β·u | β·u | **0** | 等比接收 |
| 谷区 · u<0 | β·u | **0** | β·u | **0（不动）** |

三点读数：

1. **floor 与 supn8 是互补的单侧裁剪**：floor 只关"谷区·负向"一格；supn8 关掉整列正向
   （两格）。二者关掉的格子互不重叠，不是同类改法。
2. **ST ≈ floor ∩ supn8 的守恒版**：唯一主动力源只剩"头部·负向"，其余被动接收或不动。
   ST 的 donor 定义恰是这两条裁剪线的交集。
3. **阈值是同一条线**：floor 谷区判据 `p⁺ < 0.1·max`，ST 的 `head_ratio` 也是 0.10 ——
   floor 的"谷区"字面上就是 ST 的"非头部"（就 p⁺ 维而言）。差异在 ST 额外要求学生头部。

**两个实质差异（超出格子层面）**

- **头部用谁判定**：floor 只看 p⁺；ST 看 p⁺ ∩ 学生（即第三节所述对 Ren 变量的修正）。
- **正向抬升的命运**：floor 完整保留 β·u 的正向指数抬升（连谷区正向也抬）；supn8 全删；
  ST 改为等比接收。而 supn8 已证"全删正向 → 终点与 A 持平"，故 floor 的（可能）增益必定
  来自"不压谷区负向"这一格，与正向无关。

---

## 七、被 floor 截断的量有多大

`floor_clamped_frac` 稳定在 **0.60-0.65** —— 即**约六成的 support 维**在每个位置被截断。
这不是边角修正，是对目标构造的大幅改写。参照：同期 ST 臂的 `st_donor_tokens_frac` 仅 ~0.4%
（donor 极稀疏），`st_delta_tv` ≈ 0.059；floor 改写的维数量级高两个数量级，但改的是"不施力"
而非"移动质量"，两者的 TV 影响不能直接比。

---

## 八、归因缺口（诚实边界，使用本档案时必读）

ahafloor6k 的 TB 51.60 **不能**直接作为"floor 算法优于原版 A"的证据，因为它同时混杂三个变量：

1. **floor 算法**（谷区负向截断）
2. **非 pair 视图**（单图对比，u 语义不同）
3. **102 步 = 2 epoch**（pair 系列全部只跑 51 步 = 1 epoch）

而 **非 pair + 6karmA 的纯 A 基线不存在** —— 该视图下只跑过 floor、center、center+floor 三个
变体。最接近的同视图对照是 center（TB 峰 49.14），但 center 本身是另一个机制，不是纯 A。

此外：

- **V\* 上 floor 不占优**：91.62 = 我们自训 A 的 91.62，且低于官方 pair ckpt 的 93.72。
  按"V\* 为主判据"（±0.7 稳定）与"TB 均值差 <2 / 单步 <3 不可读"的判读规则，
  TB 的 +2.2 刚过边缘，V\* 的 0 则明确无增益。
- **pair 体制下的 floor 已跑过但缺 V\***：`pairfloor6k` TB 峰 51.36@20，末端塌至 47.16
  （比 pair 原版末端 49.88 差）；**V\* judge 从未做** → 这是当前最便宜的补测（ckpt 在手）。
- **理论依据在本体制不成立**（第三节）：两臂均零挤压，故"防挤压"不能解释增益。

**要把 floor 的价值做实，最小充分设计**：pair 视图 + floor-only + 51 步 + 同 seed/同 n，
与 A-n8 / supn8 / stn8 同体制 —— 四臂即构成完整的象限消融，可把三个变化拆开归因。
实现成本：在 OPD-Aha-sup 仓库加一行 clamp（与现有 `counterfactual_u_clip_pos` 同一分支点）。

---

## 九、相关文件

- 实现：`verl/trainer/ppo/core_algos.py::reconstruct_aha_target`（`:1214`，floor 分支 `:1299`）
- 单测：`opsa/aha/test_reconstruct_aha.py`
- launcher：`opsa/aha/run_aha_4b.sh`（`AHA_FLOOR_ALPHA=0.1` 切 floor 臂）
- pair+floor：`opsa/aha/train_pairfloor.sbatch`（`EXPERIMENT_SUFFIX=pairfloor6k`）
- 数据生成：`opsa/aha/prep_null_images.py`（整图 mean-RGB）、`opsa/aha/prep_pair6k.py`（pair 视图）
- 训练日志：`opsa/logs/train_floor_6karmA_ep2.log`
- 当期终审：`docs/09_08_aha_verdict.md`（repro vs floor 两臂）
- 后续臂（另一仓库 `OPD-Aha-sup`，基线 483d70f）：supn8（`counterfactual_u_clip_pos`）、
  ST（`counterfactual_st_enable`）
