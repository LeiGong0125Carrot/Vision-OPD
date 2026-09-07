#!/bin/bash
# 在 interactive session 上直接跑 V0-6karmA-pre 的评测全链
# (复用 56 流水线: 训练完成度检查会跳过训练 -> 合并11个ckpt -> TreeBench+V* -> gpt-oss判卷 -> 汇总)
# 用法: bash 63_eval_armA_pre.sh
set -uo pipefail
export SA_ENABLE=False EVT_ENABLE=False
export EXPERIMENT_SUFFIX=6karmA-pre
export TRAIN_BATCH_SIZE=48
export TASK_TRAIN_FILE=/sfs/weka/scratch/nkw3mr/Vision-OPD/data/TreeVGR-RL-37K/train_6k_armA_hide.parquet
cd /sfs/weka/scratch/nkw3mr/Vision-OPD-setup
exec bash 56_evt_pipeline.sbatch
