# 高清 6karmA 全臂评测汇总 (2026-09-10)

数据 = 标准训练集: 2459 题高清 6karmA (框尺度过滤的 Vision-OPD-6K 子集), 全臂同预算
(batch 48 × n2, 51 步 = 1 epoch, lr 2e-6, 2×RTX Pro 6000, 61 配方)。
基座 = Qwen3.5-4B (学生与教师同权重, 教师为冻结初始副本)。
评测 = TreeBench 405 题 + V* 191 题, gpt-oss-120b 判卷 (judge 口径)。

## 一、总表 (峰值)

| 方法 | TB 峰 | V* 峰 | 口径 | 轨迹形态 |
|---|---|---|---|---|
| base (不训练) | 46.91 | 83.25 | judge | — |
| 4B+hide 特权推理 (教师读出带) | 51.36 | — | judge | 参照上界 |
| V0 (标准特权 OPD) | **50.12 @35** | 未测 | judge (已补判, 注1) | 缓升 |
| Aha+floor | 51.60 @50 | 89.53 @50 | judge | 上行, 双峰同点 |
| Aha+center | 49.14 @40 | 90.05 @45 | judge | 缓升 |
| **Aha+pair** | **52.35 @35** | **93.72 @45** | judge | **双基准全场新高** |
| Aha+center+floor | 评测排队 (job 19561300) | — | — | 训练已完 51/51 |

**pair 突破 hide 读出带**: TB 52.35 > 4B hide 特权推理带 (51.4-51.6), 逼近 9B 带 (53.33);
V* 93.72 对 base +10.47 — hide 在 V* 域历史上是负资产 (框尺度轴), pair 构造
(全图+GT crop vs 全图+空白块) 按构造解决框尺度问题, u_mean≈0 (唯一无视野错配的臂)。
注: pair 教师形态 (全图+crop) 的自身读出带未测, 预计 ≥53 — 学生大概率仍在其教师带内,
"超越"只相对 hide 形态成立。待办: base + 双图 pair 形态直推对照。

**泄露审计 (2026-09-10, 用户质询后执行)**:
① 评测卫生: pair 答案文件 privilege=none, 模型路径正确 ✓;
② TreeBench: 训练 2459 图 vs TB 395 图精确 MD5 比对**零重叠** — 52.35 干净 ✓;
③ V*: **3/191 基准图与训练图字节级相同** (sa_34785/9957/27986 ↔ V* 180/113/101,
  同源 SA-1B 去重不彻底, 影响所有臂)。但该 3 题在 pair@step5 (训练未生效) 即已全对
  → 对增益贡献 ≈0; 理论最坏界 1.57pp << 实测 +10.5。结论: V* 增益非泄露驱动,
  报告时应披露此 3 题并可给出剔除口径。

注1: V0-6karmA 判卷已补 (job 19554966, 2026-09-10 晚): **judge 峰 50.12@35**, 全轨迹
46.67/48.89/47.90/46.67/48.64/**50.12**/49.38/48.15/48.64/49.88 (step10→51)。
最终公平差距 (judge vs judge): pair +2.23, floor +1.48, center −0.98 (对 V0 峰)。
(历史遗留: 此前锚点"49.14@35"为规则口径。)

注2: Aha-repro (忠实复刻, 无 floor) 未在高清数据上训练 — 4k 证据 (TB 全程 ≤ base,
单调下漂) 已足够判死原版, 见 docs/09_08_aha_verdict.md。

## 二、逐步轨迹

### TreeBench (judge 口径; V0 列为规则口径)

| step | V0(规则) | Aha+floor | Aha+center | Aha+pair |
|---|---|---|---|---|
| 5 | 44.94 | 48.40 | 46.91 | 47.65 |
| 10 | 46.17 | 46.67 | 46.91 | 47.41 |
| 15 | 48.15 | 48.64 | 45.19 | 46.67 |
| 20 | 47.65 | 47.16 | 46.67 | 49.63 |
| 25 | 46.42 | 49.88 | 48.15 | 47.41 |
| 30 | 48.40 | 49.88 | 48.40 | 48.89 |
| 35 | 49.14 | 48.15 | 47.16 | **52.35** |
| 40 | 48.89 | ⚠️38.77 | 49.14 | 50.12 |
| 45 | 48.15 | 50.86 | 47.90 | 50.12 |
| 50 | 47.90 | **51.60** | 48.40 | 50.37 |
| 51 | 49.38 | 50.12 | 48.40 | 49.88 |

### V* (judge 口径; V0 无 V* 评测)

| step | Aha+floor | Aha+center | Aha+pair |
|---|---|---|---|
| 5 | 84.82 | 82.20 | 82.72 |
| 10 | 85.34 | 83.25 | 85.34 |
| 15 | 87.96 | 85.34 | 84.29 |
| 20 | 87.43 | 85.86 | 90.05 |
| 25 | 87.43 | 86.39 | 90.05 |
| 30 | **89.53** | 89.01 | 92.67 |
| 35 | 86.91 | 89.53 | 93.19 |
| 40 | ⚠️71.20 | 89.01 | 91.62 |
| 45 | 88.48 | **90.05** | **93.72** |
| 50 | **89.53** | 87.96 | 90.05 |
| 51 | 88.48 | 89.01 | 92.67 |

pair 轨迹注: TB 峰在 @35 后小幅回落至 ~50 平台 (仍高于 floor 除峰外的全部点);
V* 自 step20 起稳居 90+ 平台, 峰 93.72。pair 晚期无深谷、无循环发作。

⚠️ floor step40: 双基准单点塌 (38.77/71.20)。已诊断为**瞬态循环发作并自愈**: 该 ckpt
405 条回答中 38 条陷入重复循环 ("within the red box ..."), step35 仅 3 条、step45 恢复
1 条。与 OPSA 崩溃对照: OPSA 进入循环即不可逆; floor 拦截谷区负压后循环吸引子失去
自增强通道, 成为可逆瞬态。部署选 ckpt 避开 step40。

## 三、关键结论

1. **Aha+floor 双基准同点新纪录**: TB 51.60 / V* 89.53 @ step50。TB/V* 峰值张力消失
   (4k 时代 TB 峰与 V* 峰从不同 ckpt)。
2. **学生追平特权教师读出带**: 无特权学生 51.60 ≈ 4B+hide 特权推理 51.36 (带 51.4-51.6,
   含 crop/draw 形态)。蒸馏上限 = 教师读出水平而非特权信息含量; 下一带 = 9B 教师读出
   (9B+hide ≈ 53.4, 需跨模型蒸馏或 9B 自蒸馏)。
3. **floor 臂是唯一"训练越久越好"的臂** (48.4→51.6 上行; V0/4k 各臂均早峰后衰),
   epoch 末仍在上行 → 多 epoch 有明确动机。
4. **center 拿下全场最佳 V\* 90.05@45** (+6.8 对 base), TB 49.14 平 V0 规则峰。
   center 的训练期签名: u 加权均值精确归零, 无词汇漂移爬升 (floor 臂 u_mean −1.1→−0.45
   的爬升在 center 臂不出现)。
5. 高清体制 u_mean = −1.0~−1.1 (4k 为 −0.4): hide 视野错配通道随上下文丰富度放大;
   center/pair 对照在高清更有判别力。
6. 4k 对照 (docs/09_08_aha_verdict.md): repro 全程 ≤ base; center 止漂无收益;
   floor 4k 峰 48.89 — 收益随数据体制放大 (48.89 → 51.60)。

## 三·五、OPSA 对照 (负结果, 本方向的动机来源)

### 方法简介 (arXiv 2608.31046)

零监督的 token 级负压方法, 与蒸馏系完全不同的路线:
- **无教师、无特权、无 reward**。每个训练批内, 把所有 rollout token 按 logp 排序,
  选出**最低 20%** 的 token;
- 被选 token 按熵排名赋**负 advantage ∈ [−1, −0.5]** (熵最高→−1, 最低→−0.5),
  其余 token advantage=0; PPO clipped loss 更新。
- 立论 (文本域): 低概率 token ≈ 犹豫/虚假分支, 压掉它们促发 "aha" 式重新考虑。

### 视觉域失败结果 (三变体, 2459 高清题; 终审 docs/09_07_opsa_verdict.md)

| 变体 | 设定 | 结局 |
|---|---|---|
| v1 | per-response 配额选择 | 16-22 步崩溃 |
| v2a | 官方批级选择 (对齐官方 --disable-thinking) | 同上 |
| think | thinking 模式 | 同上 |

- 三变体**共享衰减常数 ~×0.94/步**, 无任何收益窗口: 训练期采样准确率 65%→43% 单调下滑;
  step8 熵已跌至 0.29; 终态收敛到**短确定性循环输出**吸引子。
- 5 轮独立 code review 结论: 可归因于域机制而非移植错误。
- **失败归因** (探针系列证据): ① 选错目标 — VLM 错误是"自信的无据断言"
  (轨迹结构判对错 AUROC≈0.52, 低概率 token 不携带错误信号), OPSA 压掉的是正常词汇多样性;
  ② 形式致命 — advantage 形式对谷区 token 施全量负梯度, 触发 Ren 挤压
  (视觉分布尖: 57% 位置 top-1>0.9, 衰减比文本域快 ~20×); ③ base 模型零循环 —
  循环是训练激发的相变。
- 与本页方法的关系: floor = "别压谷区"的防御面; 选择性 β 重建 (待办⑥) =
  "只压自信无据头部"的进攻面 — 两者都是 OPSA 验尸的直接产物。

## 三·六、真实失败 case: 训练激发的循环退化

**来源**: Aha+floor@6karmA **step40** checkpoint (瞬态发作期), TreeBench index 22,
类别 Reasoning/Perspective Transform。405 题中 38 题呈此形态; step35 仅 3 题,
step45 恢复至 1 题 (自愈)。

**题目**: From the viewpoint of the female individual seated in a wheelchair —
dressed in a light-colored floral shirt and wearing a white hat — what is the
relative direction of the signboard displaying the word "PROGRAMS"?
(A. Front left / B. Front right / C. Left rear / D. Right rear; **GT = A**)

**模型输出** (共 4846 字符, 截取):

```
To determine the relative direction of the red "PROGRAMS" signboard from the
seated woman’s viewpoint:

1. **Locate the seated woman**: She is in the red box within the red box within
the red box within the red box within the red box within the red box within the
red box within the red box within the red box within the red box within the red
box within the red box within the red box ... [重复 "within the red box" 直至
1024 token 预算耗尽, 无答案]
```

**解剖**:
- 前两句推理完全正常 (审题、开列步骤), 崩溃发生在**定位性短语**
  "She is in the red box" 处 — 空间指涉词组恰是特权蒸馏反复强化的词汇区,
  局部 2-gram 循环 ("within the red box") 一旦进入便自增强;
- 无答案输出 → 规则与 judge 双判错; 38/405 此类损失直接解释 step40 的
  38.77/71.20 双基准单点塌;
- 与 OPSA 崩溃的本质区别: OPSA 的循环终态**不可逆** (之后每步更差);
  floor 体制下这是**可逆瞬态** (step45 即恢复并创新高 50.86→51.60) —
  谷区负压被拦截后, 循环吸引子失去持续供能。

## 四、方法论摘要

### 参照协议: hide+GT 特权直推 (教师视角上界, hide 无句形态)

**构图** (`eval/treebench_probe/infer_privilege.py --privilege hide --no-priv-sentence --no-think`,
渲染函数 `hide_compose`):
1. 与原图同尺寸的**纯黑画布**;
2. 每个 GT 框按 **2.42× 外扩** (每边 ~0.71 倍框宽/高的上下文余量) 裁出区域,
   **原位贴回**画布 (绝对坐标不变, 多框各贴各位, 空间关系保持);
3. 贴回区域上**画红框**标出精确 GT 范围;
4. prompt = 原题, **不加任何特权说明句** (无句), no-think 模板。

**TreeBench 直推结果** (宽松规则口径, 405 题, 2026-09-08 统一重算):

| 模型 | base 直推 | hide无句 直推 | Δ (特权头寸) |
|---|---|---|---|
| Qwen3.5-4B | 46.91 | **51.11** | **+4.2** |
| Qwen3.5-9B | 49.38 | **53.33** | **+3.9** |

答案文件: `eval/model_answer/treebench/qwen3.5-{4b,9b}-priv-hide-nosent-nothink_answer.jsonl`。

**形态要素消融** (9B, 每项单独去除的代价): 去红框 −2.7; 不外扩 (exp1.0) −3.0;
背景留 25% 亮度 −3.2; 紧凑重排替代原位 (hide_compact) −3.5; 加特权说明句 −1.2。
即: **黑底 + 原位 + 红框 + 外扩四要素齐备才是天花板形态**; 同一份 GT 框改成
文本坐标 (box_block) 价值≈0, 反向遮挡 (inverse_hide) −19.5 (因果对照)。

用途: 该 +4 头寸是任何蒸馏方法理论上可搬运的上限参照。arm A V0 学生无特权
直推 49.14@35 (规则) ≈ 吃掉 4B 头寸的 55-60%。
(注: 总表中 "4B+hide 特权推理 51.36" 为 crop-labeled 形态的另一次测量,
与本节 hide 无句 51.11 同量级不同形态。)

### V0: 标准特权 OPD (基线)

- 学生 Qwen3.5-4B 看全图 rollout (n=2/题); **教师 = 同权重冻结初始副本**, 看 hide 特权图
  (黑底画布, GT 框内容原位可见, ~2.45× 上下文余量) + 同题 prompt, 对学生轨迹逐位置
  teacher-forcing 打分。
- 支持集 = 学生 top-100 token + 尾桶 (教师全词表 log_softmax 在学生 top-k index 上
  gather, 不重归一; 尾桶 = 1−Σtop-k 补成真分布)。
- 损失 = 对称广义 JSD (α=0.5): ½KL(pS‖m)+½KL(p⁺‖m), m=½pS+½p⁺。
- 零监督: 无答案标签、无 reward、无判卷器; 信息增量全部来自"教师看得见证据位置"。
- 一句话: **把"看见证据的自己"的分布直接蒸给"看全图的自己"**。

### Aha+center: 重建蒸馏 + 按词型中心化

在 V0 之上加两级构造 (损失路径与 V0 完全共享, 唯一变量是教师分布被替换):

1. **第二个教师视图 p⁰** = 同尺寸 mean-RGB 空白图 (null): 教师什么都看不见,
   退化为语言先验。逐位置 u_k = log p⁺_k − log p⁰_k, 度量"token k 有多依赖看见证据"。
2. **按 token 类型中心化** (本臂核心): 对每条序列、每种 token 类型, 减去该类型在全序列
   所有出现位置上的 **p⁺ 加权** u 均值。
   - 动机: hide 教师看不见上下文 → 所有场景/文风词在**每个位置**都吃恒定负 u
     (词表级偏好通道, 高清体制 u_mean −1.1), β 放大后学生被拖向"只说框内话"的词汇漂移。
     中心化把 u 分解为"词表级恒定偏好 (伪信号, 滤掉)" + "位置特异异议 (真信号, 保留)"。
   - 技术要点: 逐位置**均匀**去均值会被 softmax 平移不变性完全吸收 (空操作);
     只有按类型的**差异化**去均值才改变 q。加权用 p⁺ (梯度质量所在); 生效性读数 =
     加权均值 ≈ 0 (实测 ~1e-8)。
3. **重建目标**: log q = log_softmax(log p⁺ + β·u_centered), β=4 (论文 Eq.12 最优值);
   q 取代教师分布进 V0 的 JSD 路径 (= 论文 Eq.13)。
- 与 Aha+floor 的关系: floor 拦截**谷区**负压 (防挤压, Ren ICLR25), center 滤**词表级**
  偏置 (防漂移); 两者正交, 叠加臂 (center+floor) 训练中。

### 参照: Aha+floor (当前冠军, 详见 docs/09_08_aha_verdict.md)

repro 重建 (β=4, hide vs null) + plausibility floor: 谷区 (p⁺ < 0.1·max p⁺) 的负 u 截零,
负梯度只落教师头部。动机 = Ren 挤压理论 + OPSA 三连崩验尸 + 谷区负压暴露实测
(46-70% 谷区 token 挨压)。

## 五、工件与待办

- 代码: clone Vision-OPD-OPSA opsa 分支 (ef20ab5→1325dca); launcher opsa/aha/run_aha_4b.sh
  (AHA_FLOOR_ALPHA / AHA_CENTER_U / TASK_TRAIN_FILE 切臂)
- ckpt: checkpoints/Aha-{floor,center}-Qwen3.5-4B-6karmA/global_step_{5..51}
- 判卷: eval/judge/{treebench,vstar}/aha-{floor,center}-...-6karma-*
- 待办: ① V0-6karmA 补 LLM 判卷 (公平对比); ② pair 臂评测出分 (进行中);
  ③ center+floor 叠加臂出分; ④ 多 epoch floor (预算上限判别); ⑤ 9B+hide 53.4 锚点复核;
  ⑥ 选择性 β 重建 (核心创新方向, s_k = p_conf×(−u)₊ 判据验证先行)
