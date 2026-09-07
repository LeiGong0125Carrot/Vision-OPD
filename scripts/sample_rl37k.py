"""从 TreeVGR-RL-37K 采样 6K 训练子集 (确定性, seed=42, 按行序等比抽样以保 V*/VisDrone 配比)。
输出: data/TreeVGR-RL-37K/subset6k.parquet + subset6k_indices.json
"""
import json
import random

import pandas as pd

SRC = "/sfs/weka/scratch/nkw3mr/Vision-OPD/data/TreeVGR-RL-37K/vstar30k_visdrone6k_x1y1x2y2.parquet"
N = 6000

df = pd.read_parquet(SRC)
print(f"total rows: {len(df)}")
rng = random.Random(42)
idx = sorted(rng.sample(range(len(df)), N))
sub = df.iloc[idx].reset_index(drop=True)
sub["orig_row"] = idx
out = SRC.replace("vstar30k_visdrone6k_x1y1x2y2.parquet", "subset6k.parquet")
sub.to_parquet(out)
json.dump(idx, open(out.replace(".parquet", "_indices.json"), "w"))
print(f"saved {len(sub)} rows -> {out}")
# 粗查配比 (按行序假设前30k=V*):
n_vstar = sum(1 for i in idx if i < 30000)
print(f"采样配比: 前30k段 {n_vstar} ({n_vstar/N*100:.1f}%), 后段 {N-n_vstar}")
