"""GT 特权探针: TreeBench 的 GT bbox 以什么形态注入才能抬升 acc (OPSD teacher 选型)。

两种形态 (对照组 = 已有的 direct 结果, 标准协议: 无 system, 无 <think> 预填):
  box_block   GT 框转 [0,1000] 归一化坐标, 文本块放在 question 之后 (外部证据)
  box_prefill 同样的坐标预填为 assistant 回复的开头纯文本 (变成模型"自己说的";
              不带 <think>, 避开 8B 的预填毒点)

TreeBench 的 target_instances 只有坐标没有标签; 坐标转 [0,1000] 归一化
(Qwen3-VL 的原生坐标习惯), 文本里注明。其余 (user 后缀 / 贪心参数 / <answer>
抽取) 与 infer_staged.py 完全一致, 特权是唯一变量。

运行 (vision-opd 环境):
  python infer_privilege.py --model Qwen/Qwen3-VL-4B-Instruct --privilege box_block --limit 10
"""
import argparse
import ast
import base64
import io
import json
import os
import re

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

VOPD_ROOT = os.environ.get("VOPD_ROOT", "/scratch/nkw3mr/Vision-OPD")
TSV_PATH = os.environ.get("TSV_PATH", f"{VOPD_ROOT}/data/TreeBench/TreeBench.tsv")

USER_SUFFIX = "\nSelect the best answer to the above multiple-choice question based on the image. After the reasoning process, respond with only the letter of the correct option between <answer> and </answer>."

TAGS = ["Perception/Attributes", "Perception/Material", "Perception/Physical State",
        "Perception/Object Retrieval", "Perception/OCR",
        "Reasoning/Perspective Transform", "Reasoning/Ordering",
        "Reasoning/Contact and Occlusion", "Reasoning/Spatial Containment",
        "Reasoning/Comparison"]


def question_text(item):
    if item["category"] == "OCR":
        return item["question"]
    return item["question"] + " Options:\n" + item["multi-choice options"]


def gt_boxes_norm_str(item, w, h):
    gt = ast.literal_eval(item["target_instances"])
    nb = [[round(b[0] * 1000 / w), round(b[1] * 1000 / h),
           round(b[2] * 1000 / w), round(b[3] * 1000 / h)] for b in gt]
    return ", ".join(f"[{b[0]},{b[1]},{b[2]},{b[3]}]" for b in nb)


BOX_SENTENCE = ("The key regions relevant to this question are located at the following "
                "bounding boxes [x1,y1,x2,y2], with coordinates normalized to 0-1000: {boxes}.")

# 带标签版 (标签来自 32B 打标, eval/treebench_gt_labels.jsonl)
BOX_SENTENCE_LABELED = ("The key objects relevant to this question, with their bounding boxes "
                        "[x1,y1,x2,y2] normalized to 0-1000, are: {boxes}.")

# 视觉特权 (不含任何坐标数字, 泄漏指纹只剩 "red box" 措辞, 好过滤)
DRAW_SENTENCE = ("The key regions relevant to this question are highlighted with red "
                 "bounding boxes in the image.")
DRAW_SENTENCE_LABELED = ("The key objects relevant to this question are highlighted with red "
                         "bounding boxes in the image: {labels}.")
CROP_SENTENCE = ("In addition to the full image (shown first), close-up crops of the key "
                 "regions relevant to this question are also provided.")
CROP_SENTENCE_LABELED = ("In addition to the full image (shown first), close-up crops of the "
                         "key objects relevant to this question are also provided, in order: {labels}.")

# region = Vision-OPD 训练特权的逐字复刻: teacher 只看 "2.42x 上下文裁剪 + 内画红框",
# 不看全图; 提示句 = 训练数据的 focus 原句 (prepare_data.py 的 REMOVE_HINT)。
REGION_SENTENCE = ("Only focus on the objects inside the red bounding box in the image "
                   "to answer this question.")
REGION_EXPAND = 2.42          # teacher_images 实测外扩系数 (每边 ~0.71 倍框尺寸)

# 脚手架 (非特权! 零 GT 信息, 部署时也可用): 显式坐标系变换指引。
# 依据: 看图模型 Perspective 错题 48-54% 精确落在左右镜像选项 (随机 33%), 盲模型恰为随机
# —— 感知对了, 缺的是相机系->主体系的变换这一步。
FRAME_HINT = ("Note: if the question asks about directions from a person's or object's "
              "perspective (not the camera's), first determine which way that person or "
              "object is facing, then convert directions accordingly: when they face "
              "toward the camera, their left/right is the mirror of the camera's left/right.")


def merge_overlapping_boxes(boxes):
    """重叠框合并为并集 (HiDe merge_overlapping_bboxes 等价实现, GT 框替代注意力检测)。"""
    bs = [list(map(int, b)) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while bs:
            b = bs.pop()
            i = 0
            while i < len(bs):
                o = bs[i]
                if b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3]:
                    b = [min(b[0],o[0]), min(b[1],o[1]), max(b[2],o[2]), max(b[3],o[3])]
                    bs.pop(i); changed = True
                else:
                    i += 1
            out.append(b)
        bs = out
    return bs


def hide_compact_compose(pil_img, boxes):
    """HiDe-LPD 忠实移植 (compact_and_center_with_relative_pos 的构图段, GT 框版):
    重叠框合并 -> 消除未被任何框覆盖的行/列区间 -> 紧凑重排(保相对顺序) -> 黑底紧凑图。
    紧贴 bbox 无外扩、不画红框 —— 与原版一致; 喂模型的就是这张紧凑小图。"""
    from PIL import Image as PILImage
    img = pil_img.convert("RGB")
    w, h = img.size
    bs = [[max(0,int(b[0])), max(0,int(b[1])), min(w,int(b[2])), min(h,int(b[3]))] for b in boxes]
    bs = [b for b in bs if b[0] < b[2] and b[1] < b[3]]
    bs = merge_overlapping_boxes(bs)
    xs = sorted({v for b in bs for v in (b[0], b[2])})
    ys = sorted({v for b in bs for v in (b[1], b[3])})
    x_map, nx = {}, 0
    for i in range(len(xs) - 1):
        x_map[xs[i]] = nx
        if any(b[0] < xs[i+1] and b[2] > xs[i] for b in bs):
            nx += xs[i+1] - xs[i]
    x_map[xs[-1]] = nx
    y_map, ny = {}, 0
    for i in range(len(ys) - 1):
        y_map[ys[i]] = ny
        if any(b[1] < ys[i+1] and b[3] > ys[i] for b in bs):
            ny += ys[i+1] - ys[i]
    y_map[ys[-1]] = ny
    canvas = PILImage.new("RGB", (max(nx, 28), max(ny, 28)), (0, 0, 0))
    for x0, y0, x1, y1 in bs:
        canvas.paste(img.crop((x0, y0, x1, y1)), (x_map[x0], y_map[y0]))
    return canvas


HIDE_COMPACT_SENTENCE = ("The image has been reconstructed to show only the regions relevant "
                         "to this question, preserving their relative arrangement; irrelevant "
                         "background has been removed.")
# hide + 纯标签信息句 (无坐标无指令; 依据: 坐标是噪声/标签是信息/指令是负资产 三消融)
HIDE_LABELED_SENTENCE = ("The visible regions in the image contain the following objects: {labels}.")
# hide_full = HiDe 原版完整协议: 原图在前 + 紧凑重构图在后 (inference.py 二段 message 结构)
HIDE_FULL_SENTENCE = ("In addition to the full image (shown first), a reconstructed image is "
                      "provided that shows only the regions relevant to this question, with "
                      "irrelevant background removed and relative arrangement preserved.")


def hide_compose(pil_img, boxes, expand=None, dim=0.0, with_box=True):
    """黑底原位融合 (冠军形态), 三要素可消融:
    expand   区域外扩系数 (默认 REGION_EXPAND=2.42)
    dim      背景亮度 (0=纯黑; 0.25=保留25%亮度的场景微线索)
    with_box 是否画红框 (黑底上可见即相关, 框可能冗余)"""
    from PIL import Image as PILImage, ImageEnhance
    if expand is None:
        expand = REGION_EXPAND
    anno = draw_gt(pil_img, boxes) if with_box else pil_img.convert("RGB")
    w, h = anno.size
    if dim > 0:
        canvas = ImageEnhance.Brightness(pil_img.convert("RGB")).enhance(dim)
    else:
        canvas = PILImage.new("RGB", (w, h), (0, 0, 0))
    for b in boxes:
        bw, bh = b[2] - b[0], b[3] - b[1]
        px, py = bw * (expand - 1) / 2, bh * (expand - 1) / 2
        cx1, cy1 = max(0, int(b[0] - px)), max(0, int(b[1] - py))
        cx2, cy2 = min(w, int(b[2] + px)), min(h, int(b[3] + py))
        canvas.paste(anno.crop((cx1, cy1, cx2, cy2)), (cx1, cy1))
    return canvas


def inverse_hide_compose(pil_img, boxes, expand=None):
    """hide 的互补掩码: 涂黑 GT 外扩区域、保留其余一切 —— 构造性有害视图
    (对被遮目标的任何断言按构造即幻觉), 反向蒸馏的 teacher- 候选。"""
    from PIL import ImageDraw
    if expand is None:
        expand = REGION_EXPAND
    img = pil_img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    for b in boxes:
        bw, bh = b[2] - b[0], b[3] - b[1]
        px, py = bw * (expand - 1) / 2, bh * (expand - 1) / 2
        d.rectangle([max(0, int(b[0]-px)), max(0, int(b[1]-py)),
                     min(w, int(b[2]+px)), min(h, int(b[3]+py))], fill=(0, 0, 0))
    return img


def region_crop(pil_img, box):
    """画红框 -> 按训练系数外扩裁剪 (红框在裁剪内可见), 贴边截断。"""
    w, h = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * (REGION_EXPAND - 1) / 2, bh * (REGION_EXPAND - 1) / 2
    anno = draw_gt(pil_img, [box])
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    return anno.crop((cx1, cy1, cx2, cy2))


def to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def draw_gt(pil_img, boxes):
    """GT 框画红框: 只画 GT、无序号文字 (区别于 case study 的 annotated.jpg)。"""
    from PIL import ImageDraw
    img = pil_img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    lw = max(3, int(min(img.size) * 0.005))
    for b in boxes:
        x1, y1 = max(0, b[0]), max(0, b[1])
        x2, y2 = min(img.size[0] - 1, b[2]), min(img.size[1] - 1, b[3])
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=lw)
    return img


def crop_gt(pil_img, box, pad_frac=0.10, min_side=56):
    """带边距裁剪 + 最小尺寸保护 (区别于 case study 的零边距直裁)。"""
    w, h = pil_img.size
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    x1, y1, x2, y2 = x1 - px, y1 - py, x2 + px, y2 + py
    if x2 - x1 < min_side:
        c = (x1 + x2) / 2; x1, x2 = c - min_side / 2, c + min_side / 2
    if y2 - y1 < min_side:
        c = (y1 + y2) / 2; y1, y2 = c - min_side / 2, c + min_side / 2
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return pil_img.convert("RGB").crop((x1, y1, x2, y2))


def label_list_str(label_rec):
    labs = [l for l in label_rec["labels"] if l] if label_rec else []
    return "; ".join(labs)


def labeled_boxes_str(rec):
    parts = []
    for b, lab in zip(rec["gt_boxes_norm"], rec["labels"]):
        coord = f"[{b[0]},{b[1]},{b[2]},{b[3]}]"
        parts.append(f"{lab} at {coord}" if lab else coord)
    return "; ".join(parts)


def build(item, privilege, w, h, label_rec=None, no_image=False, pil_img=None, with_hint=False, no_sentence=False, hide_opts=None):
    hide_opts = hide_opts or {}
    qs = question_text(item)
    prefill = ""
    images = None      # None -> 原图; list of b64 -> 自定义图序列
    def add_sent(s):
        nonlocal qs
        if not no_sentence:
            qs += "\n" + s
    if privilege in ("box_block", "box_prefill"):
        if label_rec is not None:
            sentence = BOX_SENTENCE_LABELED.format(boxes=labeled_boxes_str(label_rec))
        else:
            sentence = BOX_SENTENCE.format(boxes=gt_boxes_norm_str(item, w, h))
        if privilege == "box_block":
            qs += "\n" + sentence
        else:
            prefill = sentence + " "
    elif privilege == "draw":
        gt = ast.literal_eval(item["target_instances"])
        images = [to_b64(draw_gt(pil_img, gt))]
        labs = label_list_str(label_rec)
        add_sent(DRAW_SENTENCE_LABELED.format(labels=labs) if labs else DRAW_SENTENCE)
    elif privilege == "draw_box":
        # 组合特权: 图上红框 + 文本带标签坐标, 两个通道同时给
        gt = ast.literal_eval(item["target_instances"])
        images = [to_b64(draw_gt(pil_img, gt))]
        if label_rec is not None:
            qs += "\n" + BOX_SENTENCE_LABELED.format(boxes=labeled_boxes_str(label_rec))
        else:
            qs += "\n" + BOX_SENTENCE.format(boxes=gt_boxes_norm_str(item, w, h))
        qs += " These objects are also highlighted with red bounding boxes in the image."
    elif privilege == "crop":
        gt = ast.literal_eval(item["target_instances"])
        images = [item["image"]] + [to_b64(crop_gt(pil_img, b)) for b in gt]
        labs = label_list_str(label_rec)
        add_sent(CROP_SENTENCE_LABELED.format(labels=labs) if labs else CROP_SENTENCE)
    elif privilege == "hide_full":
        # HiDe 原版双图协议: 原图 + 紧凑图
        gt = ast.literal_eval(item["target_instances"])
        images = [item["image"], to_b64(hide_compact_compose(pil_img, gt))]
        add_sent(HIDE_FULL_SENTENCE)
    elif privilege == "inverse_hide":
        # 互补掩码有害视图认证跑: 文本与 direct 逐字节相同, 零说明句
        gt = ast.literal_eval(item["target_instances"])
        images = [to_b64(inverse_hide_compose(pil_img, gt, expand=hide_opts.get("expand")))]
    elif privilege == "hide_compact":
        # HiDe 原版构图 (紧凑重排, GT 框): 单张黑底紧凑图, 无红框无外扩
        gt = ast.literal_eval(item["target_instances"])
        images = [to_b64(hide_compact_compose(pil_img, gt))]
        add_sent(HIDE_COMPACT_SENTENCE)
    elif privilege == "hide":
        # 保位置融合 (HiDe-LPD 精神): 单张黑底图, 区域原位贴回
        gt = ast.literal_eval(item["target_instances"])
        images = [to_b64(hide_compose(pil_img, gt, expand=hide_opts.get("expand"),
                                      dim=hide_opts.get("dim", 0.0),
                                      with_box=not hide_opts.get("nobox", False)))]
        if label_rec is not None:
            # 用户指定组合: hide 图(无指令句) + "对象名 at [x,y,x,y]" 信息句
            add_sent(BOX_SENTENCE_LABELED.format(boxes=labeled_boxes_str(label_rec)))
        else:
            add_sent(REGION_SENTENCE)
    elif privilege == "region":
        # 训练特权复刻: 只给区域裁剪(带红框), 不给全图; TreeBench 多框 -> 每框一张
        gt = ast.literal_eval(item["target_instances"])
        images = [to_b64(region_crop(pil_img, b)) for b in gt]
        add_sent(REGION_SENTENCE)
    elif privilege == "framehint":
        qs += "\n" + FRAME_HINT
    elif privilege == "none":
        if with_hint:
            # 悬空指令对照: 干净全图 + 训练的 focus 原句 (图上并没有红框) ——
            # 复刻 Vision-OPD student 的真实输入形态, 量化悬空指令的净成本/收益
            qs += "\n" + REGION_SENTENCE
    else:
        raise ValueError(privilege)
    content = []
    if not no_image:
        for ib in (images if images is not None else [item["image"]]):
            content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{ib}"})
    content.append({"type": "text", "text": qs + USER_SUFFIX})
    return [{"role": "user", "content": content}], prefill


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
    n_fmt = sum(1 for x in data if not re.search(r"<answer>(.*?)</answer>", x["output"], re.DOTALL))
    print(f"==> 无 <answer> 标签: {n_fmt}/{len(data)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--privilege", choices=["box_block", "box_prefill", "draw", "draw_box", "crop", "region", "hide", "inverse_hide", "hide_compact", "hide_full", "framehint", "none"],
                    required=True)
    ap.add_argument("--no-image", action="store_true",
                    help="盲臂: 不给图。none+盲=语言先验地板; box_block+盲=框文本的纯几何信息含量")
    ap.add_argument("--hide-expand", type=float, default=None, help="hide 外扩系数消融 (默认2.42)")
    ap.add_argument("--hide-dim", type=float, default=0.0, help="hide 背景亮度 (0=纯黑, 0.25=暗化)")
    ap.add_argument("--hide-nobox", action="store_true", help="hide 不画红框")
    ap.add_argument("--no-priv-sentence", dest="no_sentence", action="store_true",
                    help="视觉臂不加任何特权说明句 (纯图像消融; 输出后缀 -nosent)")
    ap.add_argument("--no-think", action="store_true",
                    help="关闭模板 thinking 模式 (Qwen3.5 默认开; 关掉后与 Qwen3-VL-Instruct 可比)")
    ap.add_argument("--with-hint", action="store_true",
                    help="privilege=none 专用: 干净图 + 训练 focus 原句 (悬空指令对照臂)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--labels-from", default=None,
                    help="GT 框标签 jsonl (label_gt_boxes.py 的输出); 提供则用带标签绑定形态")
    args = ap.parse_args()

    label_map = {}
    if args.labels_from:
        with open(args.labels_from) as f:
            for line in f:
                r = json.loads(line)
                label_map[r["index"]] = r
        print(f"labels loaded: {len(label_map)}")

    tag = args.model.rstrip("/").split("/")[-1].lower()
    suffix = (args.privilege + ("-labeled" if args.labels_from else "")
              + ("-hint" if args.with_hint else "")
              + ("-blind" if args.no_image else "") + ("-nosent" if args.no_sentence else "")
              + (f"-exp{args.hide_expand}" if args.hide_expand is not None else "")
              + (f"-dim{args.hide_dim}" if args.hide_dim > 0 else "")
              + ("-nobox" if args.hide_nobox else "")
              + ("-nothink" if args.no_think else ""))
    out_path = args.out or f"{VOPD_ROOT}/eval/model_answer/treebench/{tag}-priv-{suffix}_answer.jsonl"

    df = load_dataset("csv", data_files=TSV_PATH, delimiter="\t")["train"]
    if args.limit:
        df = df.select(range(args.limit))
    print(f"loaded {len(df)} rows; privilege={args.privilege}; out -> {out_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done[r["index"]] = r
        print(f"resume: {len(done)} rows already done")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.model)

    out_f = open(out_path, "a")
    for item in tqdm(df, desc=f"{tag}-{args.privilege}"):
        if item["index"] in done:
            continue
        pil_img = Image.open(io.BytesIO(base64.b64decode(item["image"])))
        w, h = pil_img.size
        messages, prefill = build(item, args.privilege, w, h, label_map.get(item["index"]),
                                  no_image=args.no_image, pil_img=pil_img, with_hint=args.with_hint,
                                  no_sentence=args.no_sentence,
                                  hide_opts={"expand": args.hide_expand, "dim": args.hide_dim,
                                             "nobox": args.hide_nobox})

        image_inputs, video_inputs = (None, None) if args.no_image else process_vision_info(messages)
        tmpl_kw = {"enable_thinking": False} if args.no_think else {}
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             **tmpl_kw) + prefill
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            gen = model.generate(**inputs, top_p=0.001, top_k=1, temperature=0.01,
                                 repetition_penalty=1.0, max_new_tokens=args.max_new_tokens,
                                 use_cache=True, do_sample=True)
        output_text = processor.batch_decode(
            [gen[0][inputs.input_ids.shape[1]:]],
            skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

        m = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
        ans = m.group(1).strip().upper() if m else output_text

        rec = {
            "index": item["index"], "category": item["category"], "answer": item["answer"],
            "prediction": ans, "image_size": [w, h],
            "model": args.model, "privilege": suffix, "output": output_text,
        }
        done[item["index"]] = rec
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()

    summarize(list(done.values()))


if __name__ == "__main__":
    main()
