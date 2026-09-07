"""把 TreeBench 每道题导出成独立目录, 供逐题 probing / case study。

每题一个目录 eval/treebench_cases/<idx>_<类别>_<ok|wrong>/, 内含:
  image.jpg        原图 (TSV base64 解码)
  annotated.jpg    标注图: GT 框绿色, 模型预测框红色 (带序号)
  crop_gt_<i>.jpg  每个 GT 框的裁剪
  crop_pred_<i>.jpg 每个预测框的裁剪 (按轨迹中出现顺序)
  question.md      题目 / 选项 / GT 答案 / 预测 / IoU / 框坐标
  trajectory.txt   模型完整输出 (含 <think> 推理链)
  meta.json        以上信息的机器可读版

坐标口径: 绝对像素 [x1,y1,x2,y2]; 预测框解析用官方 inference_treebench.py 同款正则。
"""
import ast
import base64
import csv
import io
import json
import os
import re
import sys

from PIL import Image, ImageDraw

VOPD_ROOT = os.environ.get("VOPD_ROOT", "/scratch/nkw3mr/Vision-OPD")
TSV_PATH = f"{VOPD_ROOT}/data/TreeBench/TreeBench.tsv"
ANSWER_JSONL = f"{VOPD_ROOT}/eval/model_answer/treebench/treevgr-7b-repro_answer.jsonl"
OUT_ROOT = f"{VOPD_ROOT}/eval/model_answer/treebench/treebench_cases"

csv.field_size_limit(sys.maxsize)


def parse_pred_boxes(output: str):
    """与官方 compute_box_iou 相同的解析: <box>[x1,y1,x2,y2]</box> 且 x1<x2, y1<y2。"""
    boxes = []
    for m in re.findall(r"<box>(.*?)</box>", output, re.DOTALL):
        cm = re.match(r"\[(\d+),(\d+),(\d+),(\d+)\]", m.strip())
        if cm:
            x1, y1, x2, y2 = map(int, cm.groups())
            if x1 < x2 and y1 < y2:
                boxes.append([x1, y1, x2, y2])
    return boxes


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1, y1 = max(0, min(int(x1), w - 1)), max(0, min(int(y1), h - 1))
    x2, y2 = max(x1 + 1, min(int(x2), w)), max(y1 + 1, min(int(y2), h))
    return [x1, y1, x2, y2]


def draw_boxes(img, boxes, color, prefix):
    d = ImageDraw.Draw(img)
    lw = max(2, int(min(img.size) * 0.004))
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = clamp_box(b, *img.size)
        d.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        d.text((x1 + lw, max(0, y1 - 14 * lw // 2)), f"{prefix}{i}", fill=color)


def main():
    recs = {}
    with open(ANSWER_JSONL) as f:
        for line in f:
            r = json.loads(line)
            recs[r["index"]] = r

    os.makedirs(OUT_ROOT, exist_ok=True)
    n = 0
    with open(TSV_PATH) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            idx = int(row["index"])
            r = recs.get(idx)
            if r is None:
                print(f"!! idx {idx} 没有轨迹记录, 跳过")
                continue
            ok = r["prediction"].upper() == r["answer"].upper()
            cat = row["category"].replace("/", "-").replace(" ", "")
            case_dir = os.path.join(OUT_ROOT, f"{idx:03d}_{cat}_{'ok' if ok else 'wrong'}")
            os.makedirs(case_dir, exist_ok=True)

            img = Image.open(io.BytesIO(base64.b64decode(row["image"]))).convert("RGB")
            img.save(os.path.join(case_dir, "image.jpg"), quality=92)

            gt_boxes = [clamp_box(b, *img.size) for b in ast.literal_eval(row["target_instances"])]
            pred_boxes = [clamp_box(b, *img.size) for b in parse_pred_boxes(r["output"])]

            anno = img.copy()
            draw_boxes(anno, gt_boxes, (0, 200, 0), "G")
            draw_boxes(anno, pred_boxes, (255, 0, 0), "P")
            anno.save(os.path.join(case_dir, "annotated.jpg"), quality=92)

            for i, b in enumerate(gt_boxes):
                img.crop(b).save(os.path.join(case_dir, f"crop_gt_{i}.jpg"), quality=92)
            for i, b in enumerate(pred_boxes):
                img.crop(b).save(os.path.join(case_dir, f"crop_pred_{i}.jpg"), quality=92)

            with open(os.path.join(case_dir, "trajectory.txt"), "w") as g:
                g.write(r["output"])

            with open(os.path.join(case_dir, "question.md"), "w") as g:
                g.write(f"# idx {idx} | {row['category']} | {'✅ 答对' if ok else '❌ 答错'}\n\n")
                g.write(f"**Q:** {row['question']}\n\n")
                if row["multi-choice options"]:
                    g.write(f"**Options:**\n```\n{row['multi-choice options']}\n```\n\n")
                g.write(f"**GT 答案:** {row['answer']}   **预测:** {r['prediction']}   **IoU:** {r['iou']:.3f}\n\n")
                g.write(f"**图片尺寸:** {img.size[0]}x{img.size[1]}\n\n")
                g.write(f"**GT 框 ({len(gt_boxes)}):** {gt_boxes}\n\n")
                g.write(f"**预测框 ({len(pred_boxes)}):** {pred_boxes}\n")

            with open(os.path.join(case_dir, "meta.json"), "w") as g:
                json.dump({
                    "index": idx, "category": row["category"],
                    "question": row["question"], "options": row["multi-choice options"],
                    "answer": row["answer"], "prediction": r["prediction"],
                    "correct": ok, "iou": r["iou"],
                    "image_size": list(img.size),
                    "gt_boxes": gt_boxes, "pred_boxes": pred_boxes,
                }, g, ensure_ascii=False, indent=2)
            n += 1

    print(f"exported {n} cases -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
