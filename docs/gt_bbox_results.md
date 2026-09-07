# GT BBox 特权信息探索:完整结果与结论

时间:2026-08-28 ~ 08-31。全部实验为 TreeBench 405 题,exact-match acc(%),贪心解码,
结果 jsonl 在 `eval/model_answer/treebench/`,脚本在 `eval/treebench_probe/`
(`infer_privilege.py` 为主力),实验矩阵详表见 `TreeVGR/08_30_plan.md`。

**核心问题**:GT bounding box 作为特权信息,以什么形态注入才能让模型答得更好?
增益的机制是什么?这决定 Vision-OPD(OPSD)的 teacher 设计。

---

## 一、素材与协议

- **GT 框**:TreeBench `target_instances`,绝对像素,405 题 833 框,**原生无标签**;
- **标签**:32B 打标(`label_gt_boxes.py` → `eval/treebench_gt_labels.jsonl`),整个
  TreeVGR 生态唯一的"带标签 GT 框";RL-37K(36,752 条,V*30K+VisDrone6.7K)自带
  `{bbox, name}`,是训练扩容的现成来源;
- **协议**:无 system、无预填;Qwen3.5 上 nothink(空 think 块)= 训练模板同构,
  为主口径;坐标注入用 norm-1000(Qwen 母语)。

## 二、文本形态:窗口规律(labeled box_block,vs 各自 direct)

| 模型 | direct 基线 | Δ(labeled box_block) | 判定 |
|---|---|---|---|
| Qwen3-VL-4B | 44.94 | +0.74 | 噪声 |
| **Qwen3-VL-8B** | 44.94 | **+3.70**(净+15) | ✅ 窗内 |
| **Qwen3.5-4B** | 46.42 | **+2.72**(净+11) | ✅ 窗内(训练底座) |
| Qwen3.5-9B | 48.89 | −0.99(净−4) | ✗ 出窗 |
| Qwen3.5-27B | 49.63 | +0.99 | 噪声 |
| Qwen3-VL-32B | 51.36 | −1.23(净−5) | ✗ 出窗 |

1. **窗口规律**:文本坐标特权只对 direct 基线 ≈47-48 以下的模型有效,跟绝对能力,
   不跟规模/代际(六点定标);
2. **有效成分是标签绑定不是坐标**:8B 拆解——裸坐标 +0.74,带标签 +3.70(约 +3.0
   来自"对象名↔坐标");盲对照(无图+框文本)38.52 vs 盲地板 38.02 → **框文本
   纯几何信息量 ≈ 0**,坐标从来不是靠"可计算性"起作用;
3. box_prefill(坐标预填为模型口吻)= 净伤害(4B −1.73),预填是分布外冲击。

## 三、视觉形态:构图定律(Qwen3.5,nothink)

| 形态 | 4B(基线 46.42) | 9B(基线 48.89) |
|---|---|---|
| draw(全图+红框+标签句) | **51.60 (+5.18)** ← 4B 冠军 | 50.86 |
| crop(原图+裁剪多图) | 51.36 (+4.94) | — |
| region(2.42×裁剪+红框,训练形态复刻) | 50.12 (+3.70) | 51.11 |
| hide(黑底原位融合,带指令句) | 48.89 | 52.10 |
| **hide 无句(纯图)** | — | **53.33 (+4.44)** ← 全场纪录 |
| hide + 带标签坐标句 | — | 51.85 |
| draw_box(红框+坐标句) | 49.38 | 50.62 |
| hide_compact(HiDe 紧凑化移植) | 47.65 | 49.88 |
| hide_full(HiDe 原版双图协议) | 47.90 | 51.11 |

**构图三铁律**(全部有双向对照支撑):

1. **视觉 ≈ 文本 2×**:draw +5.18 vs box_block +2.72(同框同标签只换形态)——红框
   免除"读数字→定位"负担,直接验证 Vision-OPD 训练特权(红框图)选型;
2. **保绝对几何 + 剥离背景**:hide(原位)双规模胜 hide_compact(紧凑化,毁真实
   距离→关系类全崩)与 hide_full(双图,原图回归稀释剥离收益)——**用户原创构图
   完胜 HiDe 原版两组件**;HiDe 紧凑化的适用边界 = V* 型单目标搜索;
3. **文字通道判决书**:指令句("Only focus...")是负资产(9B hide 删句 +1.23——
   过度聚焦伤"数第 N 个"型题);信息句(对象名标签)在 4B 是正资产(draw 删句
   −2.2);坐标句在任何视觉构图上都是噪声或双刃净负(draw_box 双规模负;hide+坐标
   句 −1.48,坐标算术忽略深度摧毁 Comparison 28→21)。**同样的信息,画进像素永远
   优于写成文字。**
4. **形态耐受有容量门槛**:4B 消化不了人工构图(黑底/多图/紧凑全部垫底,冠军是
   自然图+红框的 draw);9B 通吃(冠军是 hide 纯图)。

**视觉预算注记**:全部实验用 Qwen3.5 官方 processor 默认(4~16,384 token/图,
patch16 merge2),TreeBench 原生分辨率直进,零覆盖——53.33 是在"不省 token、
不放大分辨率"的朴素预算下拿到的,增益纯来自注意力/干扰层。

## 四、增益与代价的类别解剖(9B hide 无句 vs direct)

| 类别 | Δ | 机制 |
|---|---|---|
| Comparison | **+11**(39%→64%,全场该类最高) | 比较候选集被特权圈定,"搜索+比较"降维成"比较" |
| OCR | +6(76%→85%) | 文字周边干扰物理清除 |
| ObjRet | +2(→88%) | 多跳指称链的"找"被特权做完 |
| Perspective | +2(15%→18%) | 死区,基本不动 |
| Contact | **−3**(56%→49%) | 交界处上下文被黑底切掉(指令句版靠强制盯框保住此类) |

净翻转 +18(救回 41 / 打坏 28)。打坏的三种机制:①"数第 N 个"型题需要非 GT 对象,
被涂黑(GT 覆盖度的固有边界);②深度/场景线索被抹;③"inset 小图"误读(模型把原位
贴回区域理解成拼贴插图,22/405)——可用构图说明句修(hide v2 备选)。

## 五、机制结论(为什么有效、怎么蒸)

1. **改路径,不是改评分**(logprob 探针,三规模):固定链上特权零重评分信号(答案位
   KL≈0.0003,链级 AUC≈0.51);增益全部发生在**生成期前段 token**(KL 前 50≈0.14)
   ——特权改变链的走向,链写成后结论被自身锁死;
2. **teacher 链条干净**:9B+hide 的 211 条答对链**零坐标泄漏**(nothink 下自发写
   坐标率≈0);"red box"措辞回声(70%,来自指令句)在无句版降为自发评论(32%),
   且训练的共享句设计天然中和;53.33 冠军配置的文本 prompt 与 direct **逐字节相同**,
   特权被完全压进图像通道——与训练的换图机制(`_swap_images_in_messages`)零改动同构;
3. **think 交互**(4096 截断主导,只读完成子集):think+draw 相对同题 nothink
   +8.4(4B)/+7.7(9B),特权还提高思考收敛率——特权×自产 grounding 交互最强,
   但 25-35% 难题不收敛,think-OPD 需截断兜底;think 臂按用户指示暂停;
4. **协议毒点清单**(方法论资产):8B `<think>` 预填 EOS、32B system EOS、
   4B boxfirst 答案塌缩到 A(59%,推理与字母断裂)、9B boxfirst 18% 打框早停、
   think 反刍截断——显式编排与 nothink 部署形态不兼容;
5. **先框后推税已被代际付清**:9B boxfirst 完成子集 48.8≈direct 48.89(3-VL 代是
   稳定 −2.5~−3)——TreeVGR RL 的最后一项独特价值被 3.5 代红利覆盖;mIoU 轴:
   4B 强制 grounding 保守口径 **47.97 > TreeVGR 43.35**(出框子集 64;think 自发
   grounding 质量 66-72,16% 题自发引用)。

## 六、死区与堡垒

- **Perspective(85 题)= 特权死区**:盲答 22.4 = chance,所有看图模型 13-21 <
  chance;错题 48-54% 精确落在左右镜像选项(随机 33%,盲模型恰为随机)——**感知
  对了、缺相机系→主体系变换**,任何特权形态救不动(最好的是 region 21/85,恰回
  chance)。可行方向 = framehint 推理脚手架(代码就绪,用户暂缓);
- **Contact(41 题)= TreeVGR 堡垒**:26/41 至今无人超越(9B hide 25/41 最接近);
  hide 形态此类反而 −3,语料生成时该类换 draw 构图(不掉分)。

## 七、Teacher 配方定稿与 OPD 战略

| 配对 | teacher | student | 头寸 | 过 TreeVGR(49.63)所需转化率 |
|---|---|---|---|---|
| 9B 自蒸(保底) | 9B+hide 纯图 53.33 | 9B 48.89 | 4.44 | **17%** |
| 9B→4B 跨规模(搏强主张) | 9B+hide 53.33 | 4B 46.42 | **6.91**(容量2.47+特权4.44) | 46% |
| 4B 自蒸 | 4B+draw 51.60 | 4B 46.42 | 5.18 | 62% |

- 训练接入零改动:`teacher_image_key` 换 hide 构图,`teacher_model_source` 支持独立
  teacher 权重(跨规模);建议同时把训练数据的 focus 句按"指令句负资产"结论重审;
- 前置一步:**RL-37K/V\* 子集复测形态排序**(TreeBench 结论过户到训练分布的桥);
- 主测试套件 = TreeBench(推理考场)+ V\* Bench(感知考场,`eval/vstar.json` 已就位);
  VisDrone 是训练数据不是基准;
- SFT 不需要:grounding 格式与行为在 Qwen3.5 原生涌现(think 自发引用率 16%、质量
  72 IoU),TreeVGR 的 35K 格式教学被代际买断。

## 八、关键文件索引

- 结果 jsonl:`eval/model_answer/treebench/qwen3.5-*-priv-*_answer.jsonl`
- 探针脚本:`eval/treebench_probe/infer_privilege.py`(box_block/box_prefill/draw/
  draw_box/crop/region/hide/hide_compact/hide_full/framehint/none,`--no-think`
  `--no-priv-sentence` `--no-image` `--with-hint` `--labels-from`)
- 标签:`eval/treebench_gt_labels.jsonl`;紧凑图样例:`eval/treebench_probe/hide_images/`
- 机制探针:`eval/treebench_probe/probe_logprob.py`;总结:`eval/model_answer/treebench/treebench_probe_summary.md`
- 实验矩阵与状态:`TreeVGR/08_30_plan.md`;HiDe 源码参照:`Vision-OPD/HiDe/`
- 训练数据:`data/`(Vision-OPD-6K)、`data/TreeVGR-RL-37K/`
