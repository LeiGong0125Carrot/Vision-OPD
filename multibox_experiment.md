# 多框变体（xy_rb_mb）实验记录

2026-08-26 训练（job 18928899）/ 2026-08-27 评测（job 18955787）+ 官方 judge 复判（job 18956788）。
主实验记录见 `bbox_with_redbox_results.md`（单框 xy_rb 全程 + §7 多框摘要），本文单独把
多框变体的**设置**与**最终口径（gpt-oss-120b judge）结果**写全。

## 1. 设置：与基础 Vision-OPD 的区别

三套配置并排（基础 Vision-OPD = 仓库 `scripts/prepare_data.py` 的原版流水线）：

| | 基础 Vision-OPD（原版） | xy_rb（我们的单框） | **xy_rb_mb（本实验）** |
|---|---|---|---|
| student 图 | **红框整图**（`images/`） | 干净原图（`original_images/`） | 干净原图 |
| student 文本 | problem 原文：问题 + **"Only focus on the objects inside the red bounding box…"** + 选项 + "Answer with the option's letter" | 问题（剥掉红框提示句和作答指令）+ 单框 JSON 指令 | 问题（同左）+ **多框 JSON 指令** |
| teacher 特权（图） | **裁剪放大图**（`teacher_images/`，≈bbox 区域小图，如 1946×164） | 红框整图（与 student 原图同尺寸） | 红框整图（同左） |
| teacher 特权（文本） | 无（文本与 student 相同） | system 轮给 GT 坐标 | system 轮给 GT 坐标（同左） |
| 要求的输出 | 答案字母（自由形式） | `{"bbox_2d": [...], "answer": "X"}` | `{"objects": [{"bbox_2d", "label"}, …], "answer": "X"}`（**允许 1+ 框**） |
| 损失 | 全 token 广义 JSD（α=0.5），EMA teacher（0.05），reward-free | 同左 | 同左 |

要点：

- **基础版的特权是"裁剪放大"**（分辨率优势）；我们的探针实测它是唯一能把 teacher
  答题准确率抬起来的特权（64.2% vs 38.8%，+25.4），但其坐标在裁剪坐标系里
  （IoU≈0.001），与 bbox 输出通道不兼容——这是我们改用"坐标+红框"特权的原因。
- **我们的特权是"位置信息"**（system 坐标 + 红框），实测能把 teacher 定位 IoU 从
  0.368 抬到 0.803，但**不涨答题准确率**（三次实测 ±2 内）。设计目标是蒸馏定位能力。
- **student 侧零泄漏**：user 轮与 teacher 的 user 轮逐字节相同，坐标只在 teacher 的
  system 轮；student 看干净图（基础版 student 直接看红框图 + 文本提示，红框本身就是
  给 student 的提示，不构成师生信息差的隔离）。29 号脚本 200/200 验证：teacher 图有
  红框、student 图无、文本零坐标泄漏、红框↔GT IoU 中位 0.827。

### 1.1 Prompt 原文

student / teacher 共用的 user 轮（`<image>` + 清理后的问题 + 空行 + 格式指令）：

```
Respond with exactly one JSON object:
{"objects": [{"bbox_2d": [x1, y1, x2, y2], "label": "<name>"}, ...],
 "answer": "<letter>"}
(one entry per object relevant to the question, one or more;
 coordinates normalized to 0-1000). Output nothing else.
```

teacher 独有的 system 轮（数字照传、整句抄了会产生非法 JSON，实测复读率 0.0%）：

```
Reference annotation: the referenced object spans
x1={..} y1={..} x2={..} y2={..} in 0-1000 normalized coordinates.
```

与单框 FMT 的唯一差别就是 objects 数组：这是 Qwen 系 grounding SFT 的原生 schema，
先验上更不易漂移；GT 仍只有一个框，事后 IoU 按"预测框集合对 GT 的最大 IoU"衡量。

### 1.2 训练配置

数据 `train_xy_rb_mb.parquet` 6241 条；260 步 = 1 epoch（流式 held-out：dump 先于
update，每步的题未见过）；batch 24 × rollout n=8，lr 2e-6，SAVE_FREQ 65；
2×RTX PRO 6000，17h41m，MaxRSS 430 GiB。checkpoint 65/130/195/260（已设只读）。

## 2. 结果（官方口径：gpt-oss-120b judge，与论文同款，n=191）

判分：仓库原版 `judge_qwenlm.py` + `cal_acc.py`，judge = README 指定的
`openai/gpt-oss-120b`。**本口径下 base = 83.25%，论文 84.29% 只差 2 题，基线复现成功。**
（此前 NO_JUDGE 纯规则口径对模型不中性：judge 抬 base +4.2 但对训练后模型 ±0.5 内——
训练后输出干净、规则全能判，无翻盘空间；NO_JUDGE 的 Δ 一律偏乐观约 4 点。）

括号内 = 与 base 裸 prompt 的逐题配对 Δ；✘ = bootstrap 95% CI 不含零（显著差于 base）。

| 模型 | 总体 | direct_attr (n=115) | rel_pos (n=76) |
|---|---|---|---|
| **base 裸 prompt** | **83.2** | **84.3** | **81.6** |
| base × 多框指令 | 48.2 (−35.1✘) | 67.8 (−16.5✘) | 18.4 (−63.2✘) |
| mb-step65 裸 | 79.1 (−4.2) | 80.9 (−3.5) | 76.3 (−5.3) |
| **mb-step65 × 多框指令** | **83.2 (±0.0)** | **85.2 (+0.9)** | **80.3 (−1.3)** |
| mb-step130 裸 | 77.5 (−5.8✘) | 78.3 (−6.1) | 76.3 (−5.3) |
| mb-step130 × 多框指令 | 75.4 (−7.9✘) | 80.0 (−4.3) | 68.4 (−13.2✘) |
| e1-step130 裸（单框参考） | 73.8 (−9.4✘) | 77.4 (−7.0✘) | 68.4 (−13.2✘) |
| e1-step260 裸（单框参考） | 69.6 (−13.6✘) | 61.7 (−22.6✘) | 81.6 (±0.0) |

关键配对检验（5000 次 bootstrap）：

| 对比 | Δ | 95% CI | 判定 |
|---|---|---|---|
| mb-65 裸 − base | −4.19 | [−8.90, +0.52] | ns |
| mb-65-fmt − base | ±0.00 | [−3.66, +3.66] | **精确持平** |
| mb-130 裸 − base | −5.76 | — | ✘ 显著 |
| mb-130 − e1-130（裸，同步数） | +3.66 | [−0.52, +7.85] | ns（方向优于单框，压线） |
| **mb-65-fmt − mb-130-fmt** | **+7.85** | [+3.14, +12.57] | **✔ 伤害在 65→130 步间显著发生** |

### 2.1 训练分布上的行为（rollout 多框口径，10-step 桶 ×1920 样本）

| 步数 | answer 存在率 | 条件准确率 | 严格 MB 合规 | maxIoU | 平均框数 |
|---|---|---|---|---|---|
| 1-10 | 76.7% | 45.8% | 59.9% | 0.442 | 2.66 |
| 61-70 | 96.4% | 42.9% | 93.0%（峰） | 0.463 | 1.22 |
| 251-260 | 74.0% | 36.5% | 15.5% | 0.090 | 0.80 |

- 多框自由未被使用：框数 50 步内收敛回 ~1.2，末期 0.8。
- 格式漂移照样复发（~110 步起漂、190 后崩），时间表与单框几乎相同
  → 漂移是 reward-free JSD 的内生现象，与 bbox 格式无关。
- 答案通道比单框伤得更重（answer 存在率 96.5%→74%；单框恒 97.5-99.3%）。

## 3. 结论

1. **多框训练 65 步教会了 base 驾驭不了的格式**：base 收到多框指令会疯狂枚举
   （13.5 框/条、rel_pos 崩到 18.4），mb-step65 修成 1.4 框、答案字段零缺失——
   这是全表唯一"总体/direct/rel_pos 三项全平 base"的格子。
2. 但**最好成绩是持平，不是增益**：任何类别、任何 checkpoint 都没有显著超过 base。
3. **伤害在 65→130 步之间显著发生**（格式内 −7.85 ✔），早于格式合规率的可见崩塌；
   到 130 步两种变体都显著差于 base（mb −5.76 / 单框 −9.42）。
   多框方向性优于单框（同步数 +3.66）但未过显著线。
4. 下一步（与主记录 §6/§7.4 一致）：**早停（~65 步）+ 把蒸馏目标换到答案 token、
   特权换成 crop**——即回到基础 Vision-OPD 的特权（唯一实测能涨 teacher 答题率的），
   但保留我们的零泄漏结构；坐标系不兼容在"loss 只作用答案 token"时不再是障碍。

## 附：复现

```bash
cd /sfs/weka/scratch/nkw3mr/Vision-OPD-setup
python 25_prep_data.py --variant xy_rb_mb          # 生成数据
VARIANT=xy_rb_mb STEPS=260 TAG=vopd-xy_rb_mb-e1 SAVE_FREQ=65 \
  sbatch --mem=512G 31_train_epoch.sbatch          # 训练 (job 18928899)
sbatch 35_eval_mb.sbatch                           # 合并 + V* 双格式评测 (job 18955787)
sbatch 36_judge_gptoss.sbatch                      # gpt-oss-120b 统一复判 (job 18956788)
```

日志：`Vision-OPD-setup/logs/{vopd-18928899,evalmb-18955787,judge-18956788}.log`；
judge 结果 `eval/judge/vstar/`（旧自判备份在 `vstar_selfjudge_backup_20260827`）。
