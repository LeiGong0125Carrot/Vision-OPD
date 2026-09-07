# OPSA 三阶段实验 (arXiv 2608.31046 视觉域验证)

所有新代码集中本目录; 不改 `eval/treebench_probe/` 任何现有脚本 (只 import 复用)。
verl 侧改动 (Phase C): `verl/workers/config/actor.py` / `verl/trainer/config/actor/actor.yaml`
新增 `opsa_*` 字段, `verl/trainer/ppo/core_algos.py` 新增 `compute_opsa_loss`
(已与官方 `TreeVGR/OPSA-code/.../opsa.py` 数值对拍到 1e-7), `verl/workers/actor/dp_actor.py`
新增 opsa 分支 (跳过 teacher forward)。

## 环境

每个 shell 先:
```bash
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
cd $VOPD_ROOT/opsa
```

## 执行顺序

GPU 脚本都支持 `--gpus 2`(单命令双卡数据并行, 自动分片+合并), 按顺序一条条跑即可:
1. Phase 0 采样: 4B → 9B (各 `--gpus 2`)
2. Probe A sharpness: 4B → 9B; AIME 锚单卡插空
3. Probe B answer-noise: 9B → 4B
4. 汇总 `summarize.py` (CPU)
5. Phase C 训练 (拍板后, 双卡)

---

## Phase 0: T=1.0 采样轨迹 (A/B 共用)

冒烟 (≈2 min, prediction 应与已有 greedy 轨迹一致):
```bash
python gen_rollouts_treebench.py --model Qwen/Qwen3.5-4B --limit 3 --n 1 \
    --temperature 0.01 --top-p 0.001 --top-k 1 --out results/smoke_4b.jsonl
# 对照: head -3 $VOPD_ROOT/eval/model_answer/treebench/qwen3.5-4b-direct-nothink_answer.jsonl
```

正式 (405 题 × n=4; 断点续跑, 中断重跑同命令即可)。`--gpus 2` = 单命令双卡数据并行
(程序自动 fork 两个子进程各绑一张卡、各跑一半、完成后自动合并), 先 4B 后 9B:
```bash
python gen_rollouts_treebench.py --model Qwen/Qwen3.5-4B --gpus 2
python gen_rollouts_treebench.py --model Qwen/Qwen3.5-9B --gpus 2
```
(也可以不加 --gpus, 两个 shell 各跑一个模型。) 分片中间文件 `.shard{0,1}` 会保留用于续跑。
输出: `results/rollouts_qwen3.5-{4b,9b}_t1.jsonl` (每题 4 行, 含 token_ids)。
结束时自动打印采样 acc (4B greedy 参照 46.42 / 9B 49.14; T=1.0 会低几个点, 正常)。

## Probe A: 尖锐度

```bash
# 405×4=1620 次 forward; --gpus 2 双卡并行 ≈10–20 min / 模型
python probe_sharpness.py --model Qwen/Qwen3.5-4B --rollouts results/rollouts_qwen3.5-4b_t1.jsonl --gpus 2
python probe_sharpness.py --model Qwen/Qwen3.5-9B --rollouts results/rollouts_qwen3.5-9b_t1.jsonl --gpus 2
```

校准锚 (1.7B + AIME24, 生成+统计一体, 首跑自动下载模型和数据; ≈1h, 单卡插空):
```bash
python probe_aime_anchor.py
```

## Probe B: 答案 token advantage 噪声 (hide 特权 teacher)

学生轨迹 = Phase 0 采样 (每题自动取 1 正确 + 1 错误); H 视图 = hide_compose 黑底原位图
(hide-nosent 冠军臂, 文本与 F 逐字节相同); u = log p^H − log p^F。
```bash
# ≤810 链 × 2 forwards; --gpus 2 双卡并行 ≈15–30 min / 模型
python probe_answer_noise.py --model Qwen/Qwen3.5-9B --rollouts results/rollouts_qwen3.5-9b_t1.jsonl --gpus 2
python probe_answer_noise.py --model Qwen/Qwen3.5-4B --rollouts results/rollouts_qwen3.5-4b_t1.jsonl --gpus 2
```
冒烟: 任一命令加 `--limit 5`。

## 汇总 (CPU)

```bash
python summarize.py --source sharpness --rows results/sharpness_rows_qwen3.5-4b.jsonl
python summarize.py --source sharpness --rows results/sharpness_rows_qwen3.5-9b.jsonl
python summarize.py --source sharpness --rows results/sharpness_rows_qwen3-1.7b_aime.jsonl   # 锚
python summarize.py --source answer_noise --rows results/answer_noise_rows_qwen3.5-4b.jsonl
python summarize.py --source answer_noise --rows results/answer_noise_rows_qwen3.5-9b.jsonl
python summarize.py --source opsa_dump    # 交叉验证: 已有 RL-37K T=1.0 dump (4B)
```

判读参照 (论文数字):
- 噪声率: 4B teacher 30.6% (正确链负优势 20.4% / 错误链正优势 40.8%); teacher 越大越噪。
- 近零优势: frac(|u|≤1e-4) 全 token 51.7%; 保留 top-20% 高 logp token 时 97.5%。
- 尖锐度无绝对标尺 → 与 1.7B AIME 锚同口径对比; RL-37K 交叉验证预览 (60 链):
  mean_ent 0.60 / frac(H>1) 23% / tail-20% 熵 mean 1.36 / tail 构成 word 84% digit 3%。
- 若 TreeBench tail-20% 被坐标数字主导 → Phase C 考虑加坐标排除变体 (再议)。

---

## Phase C: 视觉域 OPSA 训练 (本 clone = Vision-OPD-OPSA, opsa 分支)

**本目录是 Vision-OPD 的独立 clone**: verl 的 SA/EVT+OPSA 改动已以 commit 形式落在 `opsa`
分支; `data` 与 `opsa/results` 是指向主仓库的符号链接; checkpoints/rollouts/merged 独立。

**零监督零特权**: teacher/ref 模块完全不加载 (main_ppo needs_ref 解耦), 不读 hide 列,
reward 恒零旁路 (ray_trainer opsa 分支), 数据只消费 prompt+原始高清全图。
配方 = 官方 run_opsa.sh 对齐: n=1 / batch 64 / lr 1e-6 constant / lowest-20% /
A∈[-1,-0.5] 熵自适应 / PPO clip / 无 IS / 无 KL。
数据 (用户拍板) = **train_6k_armA_hide.parquet 的 2459 道高清题** (只用原图, V* 同域),
高清配置对齐 Arm A: prompt 8192 / ulysses_sp 2 / activation offload / micro_bs 1。
3 epochs ≈ 115 步, save_freq 10, MAX_RESPONSE_LENGTH 2048。

干跑冒烟 (2 步, 断言看下面):
```bash
cd /sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
cd /sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA   # 00_env.sh 可能 cd 走, 回来
TOTAL_TRAIN_STEPS=2 EXPERIMENT_SUFFIX=dryrun bash opsa/run_opsa_4b.sh
```
冒烟检查项:
- 无 teacher/ref 模块加载日志, 无 `timing_s/update_actor/teacher_forward` (零特权解耦生效);
- `opsa/selected_fraction` ≈ 0.2, `opsa/advantage_mean` ∈ [-1,-0.5], loss 有限;
- `checkpoints/OPSA-Qwen3.5-4B-dryrun/` 有 ckpt。

fixed 消融干跑 (1 步):
```bash
TOTAL_TRAIN_STEPS=1 OPSA_MODE=fixed OPSA_FIXED_ADV=-0.5 EXPERIMENT_SUFFIX=fixdry bash opsa/run_opsa_4b.sh
```

正式训练 (双卡):
```bash
bash opsa/run_opsa_4b.sh
```
训练中监控 (tensorboard_log/): response_length 应缓涨不塌缩 (EVT-neg 在 6K 上崩溃的对照),
grad_norm 平稳, actor/entropy 下降但不归零 (论文 Fig 11a), selected_fraction≈0.2。
对照基线: 同数据 StdOPD-region-6karmA-pre (TB 峰 49.14 @step35); base TB 46.91 / V* 83.25 (judge 口径)。

评测: 合并+TreeBench 在本 clone 内即可; V*+gpt-oss 判卷复用主仓库
`Vision-OPD-setup/56_evt_pipeline.sbatch` 的 4–6 段 (--model 传本 clone merged 的绝对路径):
```bash
# 合并 (每个 step)
python -m verl.model_merger merge --backend fsdp \
    --local_dir checkpoints/OPSA-Qwen3.5-4B/global_step_60/actor \
    --target_dir merged/OPSA-Qwen3.5-4B-step60
# TreeBench
cd eval/treebench_probe && python infer_privilege.py \
    --model $VOPD_ROOT/merged/OPSA-Qwen3.5-4B-step60 --privilege none --no-think
# V* + gpt-oss 判卷: 同 56 流水线 4-6 段
```
参照基线 (judge 口径): base TB 46.91 / V* 83.25; V0 峰 TB 48.40 / V* 83.77。

## 文件清单

| 文件 | 用途 |
|---|---|
| `gen_rollouts_treebench.py` | Phase 0: TreeBench T=1.0 采样 (存 token_ids) |
| `probe_sharpness.py` | Probe A: teacher-force 逐 token logp/ent/top5 dump |
| `probe_aime_anchor.py` | Probe A 锚: Qwen3-1.7B + AIME24 |
| `probe_answer_noise.py` | Probe B: hide 双 forward u-dump + 答案 token 定位 |
| `summarize.py` | CPU 汇总报告 (sharpness / answer_noise / opsa_dump) |
| `run_opsa_4b.sh` | Phase C 训练启动 (官方超参对齐) |
| `results/` | 所有 dump 与轨迹输出 |
