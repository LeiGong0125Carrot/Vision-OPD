"""--gpus N 单命令多卡数据并行: 父进程按可见卡 fork N 个子进程 (各绑一张卡、各跑一个
分片、各写独立分片文件), 全部成功后自动合并到最终输出。

分片文件保留 (断点续跑靠它); 重跑同命令 = 子进程各自续跑 + 重新合并。
"""
import os
import subprocess
import sys


def _strip_args(argv, names):
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in names:
            skip = True
            continue
        if any(a.startswith(n + "=") for n in names):
            continue
        out.append(a)
    return out


def fanout_and_merge(gpus, final_out):
    """gpus<=1 时返回 False (调用方继续单卡逻辑); 否则 fork/等待/合并后返回 True。"""
    if gpus <= 1:
        return False
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = vis.split(",") if vis else [str(i) for i in range(gpus)]
    if len(ids) < gpus:
        raise SystemExit(f"可见 GPU 只有 {len(ids)} 张 (CUDA_VISIBLE_DEVICES={vis}), --gpus {gpus} 开不出来")

    argv = _strip_args(sys.argv[1:], ["--gpus", "--out", "--shard", "--num-shards"])
    shard_paths = [f"{final_out}.shard{k}" for k in range(gpus)]
    procs = []
    for k in range(gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ids[k]
        cmd = [sys.executable, os.path.abspath(sys.argv[0]), *argv,
               "--shard", str(k), "--num-shards", str(gpus), "--out", shard_paths[k]]
        print(f"[gpu {ids[k]}] {' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
    rcs = [p.wait() for p in procs]
    if any(rcs):
        raise SystemExit(f"分片子进程退出码 {rcs}; 分片文件已保留, 修复后重跑同命令即可续跑")

    with open(final_out, "w") as fo:
        for sp in shard_paths:
            with open(sp) as fi:
                fo.write(fi.read())
    print(f"merged {gpus} shards -> {final_out}", flush=True)
    return True
