"""轨迹截断续写探针: TreeVGR 轨迹截到最后一个 </box> 为止, 让 LLM/VLM 盲续写。

问题: TreeVGR 的轨迹在"最后一个框落地"时, 答案所需的信息是否已全部进入语言流?
  - 前缀 P = TreeVGR 轨迹 (treevgr-7b-repro_answer.jsonl 的 output) 截到最后一个
    </box> (含)。TreeVGR 轨迹以 "<think>" 预填开头, 续写时同样预填 "<think>" + P。
  - 续写模型不看图 (--with-image 可开对照), 输入含图像尺寸 (坐标是绝对像素)。
  - --strip-boxes 消融: 同位置截断但抹掉 <box>...</box>, 两组之差 = 坐标本身的贡献。

模型三档: 纯 LLM (Qwen3-4B-Instruct-2507) / VLM 不给图 / VLM 给图。
读数按人群拆 (结合 treebench_error_labels.json 的 error_mode):
  TreeVGR 对的 201 题 -> 盲续写保持率; ANSWER_MAP 12 题 -> 预期全救回;
  RELATION_ERR/定位对答错 -> 坐标可计算关系部分救回; REF_FRAME -> 预期救不回。

运行 (vision-opd 环境):
  python infer_traj_continue.py --model Qwen/Qwen3-4B-Instruct-2507 --limit 10
  python infer_traj_continue.py --model Qwen/Qwen3-VL-4B-Instruct --limit 10
"""
import argparse
import ast
import base64
import io
import json
import os
import re

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import torch

VOPD_ROOT = os.environ.get("VOPD_ROOT", "/scratch/nkw3mr/Vision-OPD")
TSV_PATH = os.environ.get("TSV_PATH", f"{VOPD_ROOT}/data/TreeBench/TreeBench.tsv")
TRAJ_PATH = os.environ.get(
    "TRAJ_PATH", f"{VOPD_ROOT}/eval/model_answer/treebench/treevgr-7b-repro_answer.jsonl")

MAX_PREFIX_CHARS = 4000   # 防 TRUNC 死循环轨迹撑爆前缀

TAGS = ["Perception/Attributes", "Perception/Material", "Perception/Physical State",
        "Perception/Object Retrieval", "Perception/OCR",
        "Reasoning/Perspective Transform", "Reasoning/Ordering",
        "Reasoning/Contact and Occlusion", "Reasoning/Spatial Containment",
        "Reasoning/Comparison"]


def question_text(item):
    if item["category"] == "OCR":
        return item["question"]
    return item["question"] + " Options:\n" + item["multi-choice options"]


def make_prefix(output: str, strip_boxes: bool, cut: str):
    """按切点截前缀; 无法截断或过长的返回 None (跳过)。
    first: 第一个 <box> 之前 (零坐标, 语言先验基线)
    last:  最后一个 </box> 为止 (含全部定位证据)
    full:  <answer> 之前的全轨迹 (再加全部语言推理)
    """
    if cut == "first":
        pos = output.find("<box>")
        if pos < 0:
            return None
        prefix = output[:pos].rstrip()
    elif cut == "last":
        pos = output.rfind("</box>")
        if pos < 0:
            return None
        prefix = output[: pos + len("</box>")]
    elif cut == "full":
        pos = output.find("<answer>")
        if pos < 0:
            return None
        prefix = output[:pos]
        # 保留 </think> 之前的部分, 避免把收尾语带进去
        prefix = prefix.split("</think>")[0].rstrip()
    else:
        raise ValueError(cut)
    if len(prefix) > MAX_PREFIX_CHARS:
        return None
    if strip_boxes:
        prefix = re.sub(r"\s*<box>.*?</box>", "", prefix, flags=re.DOTALL)
    return prefix


def summarize(data):
    total = correct = 0
    for tag in TAGS:
        g = [x for x in data if x["category"] == tag]
        c = sum(1 for x in g if x["prediction"].upper() == x["answer"].upper())
        total += len(g)
        correct += c
        if g:
            print(tag, f"{c}/{len(g)}={round(c / len(g) * 100, 2)}")
    print("==> Overall", f"{correct}/{total}={round(correct / total * 100, 2)}")
    keep = sum(1 for x in data if x["treevgr_correct"] and x["prediction"].upper() == x["answer"].upper())
    n_tv = sum(1 for x in data if x["treevgr_correct"])
    fix = sum(1 for x in data if not x["treevgr_correct"] and x["prediction"].upper() == x["answer"].upper())
    n_wr = sum(1 for x in data if not x["treevgr_correct"])
    print(f"==> TreeVGR对的保持: {keep}/{n_tv}   TreeVGR错的救回: {fix}/{n_wr}")
    n_fmt = sum(1 for x in data if not re.search(r"<answer>(.*?)</answer>", x["output"], re.DOTALL))
    print(f"==> 无 <answer> 标签: {n_fmt}/{len(data)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="纯 LLM 或 VLM 均可")
    ap.add_argument("--with-image", action="store_true", help="给图 (仅 VLM); 默认盲续写")
    ap.add_argument("--strip-boxes", action="store_true", help="消融: 前缀里抹掉 <box> 坐标")
    ap.add_argument("--cut", choices=["first", "last", "full"], default="last",
                    help="截断点: first=首个box前(语言先验基线) last=末个box后 full=answer前全轨迹")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    tag = args.model.rstrip("/").split("/")[-1].lower()
    variant = ("img" if args.with_image else "blind") + f"-cut{args.cut}" + ("-nobox" if args.strip_boxes else "")
    out_path = args.out or f"{VOPD_ROOT}/eval/model_answer/treebench/{tag}-trajcont-{variant}_answer.jsonl"

    trajs = {r["index"]: r for r in map(json.loads, open(TRAJ_PATH))}
    df = load_dataset("csv", data_files=TSV_PATH, delimiter="\t")["train"]
    if args.limit:
        df = df.select(range(args.limit))
    print(f"loaded {len(df)} rows; variant={variant}; out -> {out_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done[r["index"]] = r
        print(f"resume: {len(done)} rows already done")

    # 纯 LLM 与 VLM 统一加载
    from transformers import AutoProcessor, AutoTokenizer
    is_vlm = True
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map="auto", low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(args.model)
    except Exception:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map="auto", low_cpu_mem_usage=True)
        processor = AutoTokenizer.from_pretrained(args.model)
        is_vlm = False
    if args.with_image and not is_vlm:
        raise SystemExit("--with-image 需要 VLM")

    from qwen_vl_utils import process_vision_info

    out_f = open(out_path, "a")
    skipped = 0
    for item in tqdm(df, desc=f"{tag}-{variant}"):
        if item["index"] in done:
            continue
        traj = trajs[item["index"]]
        prefix = make_prefix(traj["output"], args.strip_boxes, args.cut)
        if prefix is None:
            skipped += 1
            continue
        w, h = Image.open(io.BytesIO(base64.b64decode(item["image"]))).size

        qs = question_text(item)
        ctx = (f"\n(The image is {w}x{h} pixels; object locations in the reasoning below are "
               f"given as <box>[x1,y1,x2,y2]</box> in absolute pixel coordinates.)"
               if not args.strip_boxes else f"\n(The image is {w}x{h} pixels.)")
        user_text = (qs + ctx +
                     "\nSelect the best answer to the above multiple-choice question. Continue the "
                     "reasoning below and respond with only the letter of the correct option "
                     "between <answer> and </answer>.")

        content = []
        if args.with_image:
            content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{item['image']}"})
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "user", "content": content if is_vlm else user_text}]

        prefill = "<think>" + prefix
        if is_vlm:
            image_inputs, video_inputs = process_vision_info(messages) if args.with_image else (None, None)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to(model.device)
        else:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill
            inputs = processor([text], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            gen = model.generate(**inputs, top_p=0.001, top_k=1, temperature=0.01,
                                 repetition_penalty=1.0, max_new_tokens=args.max_new_tokens,
                                 use_cache=True, do_sample=True)
        toks = gen[0][inputs.input_ids.shape[1]:]
        output_text = (processor.batch_decode([toks], skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False)[0]
                       if is_vlm else processor.decode(toks, skip_special_tokens=False))

        m = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
        ans = m.group(1).strip().upper() if m else output_text

        rec = {
            "index": item["index"], "category": item["category"], "answer": item["answer"],
            "prediction": ans,
            "treevgr_prediction": traj["prediction"],
            "treevgr_correct": traj["prediction"].upper() == item["answer"].upper(),
            "prefix_len": len(prefix), "model": args.model, "variant": variant,
            "output": output_text,
        }
        done[item["index"]] = rec
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    if skipped:
        print(f"跳过 (无框或前缀超长): {skipped}")
    summarize(list(done.values()))


if __name__ == "__main__":
    main()
