"""V* 191 题的 hide无句 头寸探针.

目的: 测高清体制(V* 图)下 hide 特权视角能否给 4B 抬分 —— 6K+hide 训练实验的免费前置判决.
  --view full  全图基线臂 (同口径重测, 对齐 judge 缓存的 83.25 参照)
  --view hide  hide无句臂 (TreeBench 61.0 冠军形态: 外扩2.42/纯黑底/画红框/无特权句/no-think)

GT 框来源: craigwu/vstar_bench 每图同名 .json ({target_object, bbox[x,y,w,h]}),
首次运行自动下载标注(仅 json, <1MB)到 eval/vstar_data/annotations/.

输出: model_answer/vstar/{tag}_answer.jsonl (07_eval 同构: 原行 + model_answer),
判卷: judge_qwenlm.py --benchmark vstar --model {tag}

用法:
  python infer_vstar_hide.py --dry-run            # CPU: 校验映射+渲染样例
  python infer_vstar_hide.py --model M --view hide
  python infer_vstar_hide.py --model M --view full
"""
import argparse
import json
import os

from PIL import Image
from tqdm import tqdm

from infer_privilege import hide_compose  # 冠军渲染, 与 TreeBench hide-nosent 完全一致

VOPD_ROOT = os.environ.get("VOPD_ROOT", "/scratch/nkw3mr/Vision-OPD")
VSTAR_JSON = f"{VOPD_ROOT}/eval/vstar.json"
ANN_DIR = f"{VOPD_ROOT}/eval/vstar_data/annotations"


def ensure_annotations():
    marker = f"{ANN_DIR}/test_questions.jsonl"
    if not os.path.exists(marker):
        from huggingface_hub import snapshot_download
        print("下载 vstar_bench 标注 (仅 json)...")
        snapshot_download(
            repo_id="craigwu/vstar_bench", repo_type="dataset", local_dir=ANN_DIR,
            allow_patterns=["direct_attributes/*.json", "relative_position/*.json",
                            "test_questions.jsonl"])
    return marker


def load_items():
    """vstar.json 行 ⟕ test_questions.jsonl (键: category+题面) -> 附 GT 框(绝对像素 x1y1x2y2)."""
    ensure_annotations()
    tq = {}
    with open(f"{ANN_DIR}/test_questions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            tq[(r["category"], r["text"].strip())] = r["image"]
    items = json.load(open(VSTAR_JSON))
    out, misses = [], []
    for it in items:
        key = (it["category"], it["query"].strip())
        img_rel = tq.get(key)
        if img_rel is None:
            misses.append(it["question_id"])
            continue
        ann = json.load(open(f"{ANN_DIR}/{os.path.splitext(img_rel)[0]}.json"))
        boxes = [[x, y, x + w, y + h] for x, y, w, h in ann["bbox"]]
        out.append({**it, "gt_boxes": boxes, "targets": ann["target_object"]})
    if misses:
        raise RuntimeError(f"映射失败 {len(misses)} 题: {misses[:5]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--view", choices=["full", "hide"], default="hide")
    ap.add_argument("--tag", default=None, help="输出文件名 tag; 默认 vstar-probe4b-{view}")
    ap.add_argument("--dry-run", action="store_true", help="CPU 校验: 映射+渲染样例, 不加载模型")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    items = load_items()
    print(f"映射完成: {len(items)} 题均有 GT 框; 框数分布:",
          {n: sum(1 for it in items if len(it["gt_boxes"]) == n) for n in {len(it["gt_boxes"]) for it in items}})
    if args.limit:
        items = items[:args.limit]

    if args.dry_run:
        outd = f"{VOPD_ROOT}/eval/treebench_probe/vstar_hide_preview"
        os.makedirs(outd, exist_ok=True)
        for it in items[:4]:
            img = Image.open(it["images"][0])
            hide_compose(img, it["gt_boxes"]).save(f"{outd}/{it['question_id']}_hide.jpg")
            print(f"qid={it['question_id']} {it['category']} img={img.size} boxes={it['gt_boxes']} targets={it['targets']}")
        print(f"渲染样例 -> {outd}/ (肉眼确认红框区域原位可见、背景纯黑)")
        return

    assert args.model, "--model 必填 (非 dry-run)"
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from infer_privilege import to_b64

    tag = args.tag or f"vstar-probe4b-{args.view}"
    out_path = f"{VOPD_ROOT}/eval/model_answer/vstar/{tag}_answer.jsonl"
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(l)["question_id"] for l in f}
        print(f"resume: {len(done)} 已完成")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)

    out_f = open(out_path, "a")
    for it in tqdm(items, desc=f"{tag}"):
        if it["question_id"] in done:
            continue
        img = Image.open(it["images"][0])
        view = hide_compose(img, it["gt_boxes"]) if args.view == "hide" else img
        content = [{"type": "image_url", "image_url": f"data:image/jpeg;base64,{to_b64(view)}"},
                   {"type": "text", "text": it["query"]}]
        messages = [{"role": "user", "content": content}]
        image_inputs, video_inputs = process_vision_info(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            gen = model.generate(**inputs, top_p=0.001, top_k=1, temperature=0.01,
                                 repetition_penalty=1.0, max_new_tokens=args.max_new_tokens,
                                 use_cache=True, do_sample=True)
        output_text = processor.batch_decode(
            [gen[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()

        rec = {"images": it["images"], "query": it["query"], "response": it["response"],
               "question_id": it["question_id"], "category": it["category"],
               "sample_uid": f"vstar:question_id:{it['question_id']}",
               "model_answer": output_text}
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    print(f"done -> {out_path}\n判卷: $VOPD_PY judge_qwenlm.py --benchmark vstar --model {tag} "
          f"--api_base http://localhost:8813/v1/ --judge_model openai/gpt-oss-120b")


if __name__ == "__main__":
    main()
