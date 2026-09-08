# OPD-Aha 自研复刻 + Plausibility Floor

复刻 `revision_opd(1).pdf` 的重建式蒸馏 (Eq.6-14), 并验证由 Ren & Sutherland
(ICLR25 学习动力学) 推出的抗挤压改进。官方 OPD-Aha 代码不可获取, 本实现走我们
自己的 verl 机器 (与 V0 标准特权 OPD **共享完全相同的损失路径**, 唯一变量是
teacher 分布被换成重建 q)。

## 方法

- teacher = frozen init copy; 两次 no_grad 前向:
  - p⁺ = teacher(hide 特权图) — 域适配: 论文用 crop, 我们用本组 TreeBench 冠军
    形态 hide;
  - p⁰ = teacher(mean-RGB 同尺寸空白图) — 论文同款 null 参照。
- 支持集 = 学生 top-100 + 尾桶; u = log p⁺ − log p⁰;
  `log q = log_softmax(log p⁺ + β·u)` (β=4, 论文最优), 即 Eq.12 的
  q ∝ softmax((1+β)log p⁺ − β log p⁰)。
- 损失 = 广义 JSD α=0.5 = 论文 Eq.13 ½KL(pS‖m)+½KL(q‖m)。
- **floor 臂**: p⁺ < α_floor·max(p⁺) 的谷区维负 u 截 0 (α_floor=0.1) —— 负梯度
  只落头部, 按 Ren 挤压理论应消除 OPSA 式崩溃触发 (实测暴露面: 谷区 token 46.4%
  吃负压)。α_floor→0 平滑退回原版。

## 文件

- `prep_null_images.py` — 生成 4000 张 mean-RGB null 图 + `train_sa4k_aha.parquet`
  (已跑过, 幂等可重跑)
- `run_aha_4b.sh` — 训练 launcher (fork 自 scripts/run_state_adaptive_4b.sh)
- `test_reconstruct_aha.py` — `reconstruct_aha_target` CPU 单测 (已过)

## 代码改动 (本 clone, opsa 分支)

| 文件 | 改动 |
|---|---|
| `core_algos.py` | `reconstruct_aha_target()` — 重建 + floor, no_grad 纯函数 |
| `config/actor.py` + `actor.yaml` | `aha_enable/aha_beta/aha_floor_alpha/null_image_key` + 互斥/前置校验 |
| `ray_trainer.py` | teacher 输入构建抽成 `build_side()`, aha 时对 null_images 再跑一次; `_get_gen_batch` 白名单加 null 列 |
| `dp_actor.py` | select_keys 加 null_*; 第三个 no_grad teacher 前向 (计时键 `teacher_null_forward`); q 替换 teacher_topk_logps 后走原 V0 损失 |

## 运行

```bash
# 臂 1: Aha-repro (忠实复刻, β=4)
bash opsa/aha/run_aha_4b.sh

# 臂 2: Aha+floor (抗挤压)
AHA_FLOOR_ALPHA=0.1 bash opsa/aha/run_aha_4b.sh
```

同 V0 预算: train_sa4k_aha 4000 行, batch 96 × n 8, lr 2e-6, 1 epoch ≈ 41 步,
save_freq 5。每步比 V0 多一次 teacher 前向, 预计 6-9h/臂。

干跑冒烟 (2 步) 断言:
1. wandb/console 出现 `timing_s/update_actor/teacher_forward` **和**
   `teacher_null_forward` 两个计时键且都增长;
2. `aha/u_mean`、`aha/q_tail_mass` 有限; loss 有限;
3. repro 臂 `aha/u_neg_valley_frac` > 0; floor 臂 `aha/floor_clamped_frac` > 0;
4. `aha/argmax_conf` (挤压哨兵, Ren Fig.4): repro 臂若快速上行 → 挤压再现;
   floor 臂应平缓。

## 评测与判读

eval_opsa.sbatch 模式 (合并 → TreeBench → V* → gpt-oss 判卷)。参照 (judge 口径):
base TB 46.91 / V* 83.25; V0 峰 48.40 / 83.77。

- Aha-repro > V0 → 重建机制在我们复刻下成立;
- (+floor) − (repro) → 挤压保护的净贡献;
- 哨兵曲线 (argmax_conf / 反思率 / 长度) 给机制注脚。
