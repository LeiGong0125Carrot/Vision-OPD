# OPSA 视觉域验证终审报告 (2026-09-06 ~ 09-07)

**论文**: Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement
(arXiv 2608.31046, Purdue; OPSA = On-Policy Self-Adaptation)

**结论一句话**: 论文的"teacher 可去"论断在视觉域复现且更强；但去掉 teacher 之后的 OPSA
机制本身在视觉域**不成立**——三种设置（per-response 配额 / 官方 batch 级语义 / 思考模式
域扩展）共享同一条熵衰减曲线（~×0.94/步，比论文数学域快 ~20 倍），16-22 步内均崩溃，
且**不存在崩溃前的收益窗口**（on-policy acc 从 step 1 起单调下降）。实现经两轮共五份
独立 code review 背书：与官方逐 token 对拍 <1e-7，运行时配置逐项对齐官方配方。

---

## 1. 背景与动机

OPSA 论文两大主张：
1. OPD 的 teacher 监督噪声大且学生对其不敏感——收益不来自蒸馏，而来自"压制学生自采的
   低 logp token"；
2. 因此可以完全去掉 teacher：在 batch 级 lowest-20%-logp token 上施加熵自适应负 advantage
   （[-1,-0.5]，选中集内熵 rank，高熵→-1），零监督自提升（数学域 Qwen3-1.7B/4B/Qwen3.5-9B
   上 Avg@32 +166%~307%）。

对 Vision-OPD 主线的意义：若主张 1 在视觉域成立，特权蒸馏（SA-OPD/EVT）的监督价值存疑；
若主张 2 成立，我们有一个零成本的强 baseline。因此先 probe 验证前提，再训练验证方法。

## 2. Probe 结果 (Qwen3.5-4B, TreeBench 405 题, T=1.0 采样 405×4 链)

### 2.1 尖锐度 (probe_sharpness)

- 全序列熵 mean 0.491（p50=0.230, p99=2.33）；57.3% token top1>0.9；19.2% 位置 H>1
- tail-20%（OPSA 作用区）熵 mean 1.27；其中 64.3% 高熵（真不确定）、仅 0.5% 低熵——
  机制前提"压制落在真分叉上"成立
- tail 构成：word 84.3% / 标点 12.8% / 数字 2.9% / 答案字母 0%——**坐标 token 主导的担忧
  排除**；EOS 的 logp 分位 0.99，初始几乎不进 tail
- 高熵位 top-5 含经典反思词仅 9.4%（数学域的 wait/but 在视觉域对应物是空间关系词
  near/behind/left/visible）

### 2.2 答案 token 噪声 (probe_answer_noise, hide 特权 teacher, u = log p^H − log p^F)

| 指标 | 视觉域 (4B) | 论文 (数学域 4B teacher) |
|---|---|---|
| 正确链拿负优势 | 16.1% | 20.4% |
| 错误链拿正优势 | **59.2%** | 40.8% |
| 总噪声率 | **38.3%** | 30.6% |
| 答案 token mean u | **≈ 0**（-0.000 / -0.003） | — |

- 近零优势 frac(|u|≤1e-4)=23.4%（论文 51.7%），但结构一致：top-20% 高 logp token 上 92.7%
  近零（论文 97.5%）——有效学习信号集中在尾部
- tail-20% 上 teacher 意见：80.2% neutral / gems 4.1% / confirm 15.7%——**盲压相对特权蒸馏
  几乎不损失信息**

**Probe 判决**：主张 1 在视觉域复现且更强（hide 特权推理时 +4~5 pt acc，但 logp 级监督
净信号≈0 且 59% 错误答案被正向强化）。绿灯放行训练。

## 3. 训练三连崩溃

实现：verl 移植与官方 slime `compute_opsa` 逐 token 对拍（差 <1e-7）；零监督零特权
（无 teacher/ref 模块、reward 恒零、不读 hide 列）；三份独立 code review 确认无数值 bug。
数据 = 6K armA 高清 2459 题（只用 prompt+原图）；官方超参 n=1 / batch 64 / lr 1e-6 constant /
frac 0.2 / A∈[-1,-0.5] / PPO clip / betas(0.9,0.98) / wd 0.1（v2a 起）。

### 3.1 三个变体与三种死法

| 变体 | 与官方的差异 | 崩溃步数 | 死法 |
|---|---|---|---|
| **v1** | 选择在 micro-batch 内（micro_bs=1 → 每条 response 强制 20% 配额，无"自信免压"逃逸阀） | ~20 | 锐化→残余熵积到答案/EOS→EOS 被压→跌进 nothink 模板的 `</think>` 先验→**复读循环顶死 2048**（step20 63/64 循环，熵 0.001） |
| **v2a** | 无（官方 batch 级语义 + 官方超参；review 后的忠实还原） | ~16 | batch 池引入**隐式长度惩罚**（谁长谁挨压）→塌至 2-token→**常数策略不动点**（step19 64/64 输出同一字母 "D"，熵 7e-7，梯度 ∝(1-p)→3e-4 自熄灭） |
| **think** | 开启思考（经 Qwen 官方模板机制；注意 **OPSA 官方配方本身是 --disable-thinking**，故 v2a 才是与官方配方完全对齐的一侧——think 是我们的单变量域扩展实验，动机为论文 Table 3 thinking 变体与深熵池假说） | ~20+（观测至 step 38 熵 0.04） | 1500-token 真实推理块→**模板结晶**（"1.Locate 2.Examine 3.Compare"固定骨架）→反思词先剪除后困入**搜索循环**（"Let me look at the sign..." 复读）→截断洪水启动（clip 0.30@13） |

### 3.2 同一衰减常数（核心证据）

三变体的每 token 熵衰减逐步重合（~×0.94/步）：

| step | v1 | v2a | think |
|---|---|---|---|
| 1 | 0.566 | 0.552 | 0.532 |
| 5 | 0.489 | 0.465 | 0.437 |
| 8 | 0.330 | 0.294 | 0.340 |
| 10 | 0.118 | 0.081 | 0.195 |
| 14 | 0.028 | 0.039 | 0.110 |

论文数学域同款衰减走了 300-500 步（0.25→0.05）。**衰减速率与选择粒度、响应格式、熵池
深度（think 长度 10 倍）全部无关**——由 lr × advantage 幅度 × 压制比例决定；格式先验只
决定死法。深熵池不减缓衰减，只是"死时拖着更多推理文本"。

### 3.3 无收益窗口（think run 的 on-policy acc）

| step | 1 | 5 | 9 | 12 | 17 |
|---|---|---|---|---|---|
| acc | 65% | 61% | 55% | 52% | **43%** |

从第一步起单调下降 22 pt——**不存在"先涨后崩"，早停无法抢救**。

### 3.4 反思词 V 型反转 = "fake aha" 陷阱指标

think run 反思词计数：1004 (step1) → 699 (step9, 真反思被剪) → **1841** (step17, "Wait, let
me look again" 变成循环填充物)。若按论文的反思 token 计数评估法，step17 会被误判为
"反思推理大增"。机制不对称性：论文的反思增益依赖反思词**在 head 候选中未被采样**、等着
接收尾部回流的概率质量；视觉域反思词稀有（9.4%），一旦被采样即是低 logp token → 进选择
集 → 吃 -1.0 最强压制。**同一算子，反思词在 head 是升它、在 tail 是杀它——方向由域的
词频结构决定，"促进反思"不是方法性质。**

### 3.5 verl 语义的放大器（非崩溃根因，但加速终章）

截断链（无 EOS）在 verl 的 response_mask 中全额有效——一条 2048/4096-token 复读链的
权重是正常短回答的 ~100 倍。v1 从 19% 截断到 100% 只用 3 步即此正反馈。官方 slime 同样
不剔除截断链，但数学域长回答 EOS 充足未暴露此通路。

### 3.5b 循环考古补充（09-07 晚）

训练前循环率严格为 0（base 4B T=1.0 共 1812 链、含 2048/4096 预算，无一循环）——复读
循环是**训练激发的相变**（v1: step15 0% → step20 61%），非既有倾向的放大。三 run 的
死法精化为"**两条路径、一个终点、一次中途爆燃**"：v2a 直落短答吸引子（零循环）；think
短暂闪燃（1-3%）后同样滑入缩短终态（step38 长度 100-300）；v1 在熵地板+截断反馈+格式
先验三条件齐备时爆燃循环。推论：崩溃模型的反思词暴涨 100% 为训练制造（fake-aha 加固）。

### 3.6 终态审查背书（2026-09-07，两份独立复审）

- **通路审查**：官方八段数据通路（rollout → no-grad 重算 forward → batch 级选择 →
  advantage 入 batch → 切 micro → clipped loss → 逐样本 selected-mean 归一 → 单步优化）
  逐段等价；`compute_opsa_selection` 与官方 `compute_opsa` 逐行同构（含 tie-break、1e-12
  退化分支）；driver/worker response_mask 为同一张量、balance 重排先于选择、无陈旧
  advantage。残余偏差（clip_high 0.3、全局 vs DP-local 池）在本配置下均为惰性。
- **配置审计**（读实跑日志的 runtime dump，非推导）：lr/warmup/betas/wd/grad_clip/
  entropy_coef/KL/normalize/frac/adv 范围/n/batch/单步优化/采样参数逐项与官方一致；
  think 模板通路端到端验证（rollout input 尾部 `<think>\n`）。
- **运行事实备案**：v2a 非正常终止于 step 24（Ray ActorUnavailableError，与 think 首次
  启动的 GPU 冲突同时刻）；崩溃在 step 16-19 已完整记录，ckpt 10/20 在册，结论不受影响。
- 双审结论原文："用这份代码得到的崩溃/无效结果可以归因于视觉域机制，而非移植失真。"

## 4. 总结论

1. **主张 1 成立（且更强）**：特权 teacher 的 token 级监督在视觉域净信号≈0、噪声 38%、
   错误链 59% 被正向强化——特权蒸馏（OPD/OPSD 形式）路线的监督价值被直接质疑。
   与既有记录一致：官方 Vision-OPD-9B judge 口径与 base 打平（零迁移）。
2. **主张 2 不成立（普适性被否）**：OPSA 的目标函数是纯 mode-seeking，其全局吸引子是
   确定性策略（advantage 恒负 + 梯度 ∝(1-p) 在 δ 分布处自熄灭）。论文的稳定性与收益是
   从数学 CoT 的深熵池 + 反思词 head 分布**借来的域性质**，不是方法自带的。视觉短答案域
   两个条件都不满足 → 16-22 步崩溃、全程纯伤害。
3. 与 EVT-neg 记录（advantage 形式 0 胜 4 负、跨体制崩溃）合并为总结论：**纯负 advantage
   的压制形式在视觉（短答案/浅熵池）体制下结构性不稳定**——无论信号来源是特权残差
   (EVT-neg) 还是熵 rank (OPSA)，无论粒度、格式、超参如何对齐官方。
4. 对 SA-OPD 主线的含义：(a) 零监督 baseline 这条捷径不存在；(b) 任何借鉴"尾部压制"的
   设计必须内置熵保护（这已是改算法本体）；(c) 评估反思/长度类指标时警惕循环污染
   （fake-aha 陷阱）。

## 5. 资产清单（复现指引）

- **代码**: clone `/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA`（`opsa` 分支）。关键 commits:
  `9af8329` 移植+零特权解耦 / `730c507` _update_teacher 修复 / `3bd3227` v2a batch 级选择 /
  `8eaf2fe` THINK_MODE。verl 侧: `core_algos.compute_opsa_selection/compute_opsa_loss`、
  dp_actor opsa 分支、ray_trainer driver 级选择与 reward 旁路、main_ppo needs_ref 解耦。
- **probe**: 主仓库 `opsa/`（gen/probe_sharpness/probe_answer_noise/probe_aime_anchor/
  summarize + `--gpus N` 多卡分片）；数据 `opsa/results/*.jsonl`（rollouts/sharpness/answer_noise）。
- **训练产物**: clone 下 `checkpoints/OPSA-Qwen3.5-4B{,-v2a,-think}/global_step_*`、
  `rollouts/OPSA-Qwen3.5-4B*/N.jsonl`（逐步 rollout 文本，本报告文本证据来源）、
  `opsa/logs/`、tensorboard/wandb (project Vision-OPD-StateAdaptive)。
- **评测**: `opsa/eval_opsa.sbatch`（合并→TreeBench→V*→gpt-oss 判卷→汇总）已备好未运行
  （崩溃 ckpt 全评无意义；如需给报告补 judge 口径数字，对 step10 ckpt 单点评测即可）。
- **对照基线** (judge 口径): base TB 46.91 / V* 83.25；V0 峰 48.40/83.77；同数据
  StdOPD-region-6karmA-pre TB 峰 49.14@step35。
- 三份 code review 报告结论要点已并入本文 §3；审查范围：数学语义（官方对拍）、工程接线
  （温度/ratio/归一化/SP 梯度）、数据与模板（`</think>` 溯源、截断 mask 语义）。
