#!/usr/bin/env python3
"""
Step 1 - convert the competition CSV labels to YOLO format and build the robustness-check data.

Input  (competition kit folder): train_labels.csv, val_labels.csv, train_images/, val_images/
Output (--work folder):
    yolo/images/{train,val}   symlinks (or copies) of the images
    yolo/labels/{train,val}   one .txt per image: "cls cx cy w h" normalised, empty file = background image
    road.yaml                 dataset file used for training
    yolo_degraded/ + road_degraded.yaml + degraded_val_log.json   val copy with blur/dark/noise/JPEG damage
    val_600.txt, val_720.txt, val_other.txt + road_val_*.yaml     val subsets per image size

Usage:  python src/prepare_data.py --kit /path/to/kit --work ./work
"""
import argparse, json, os, shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

NAMES = ["D00", "D10", "D20", "D40"]
SEED = 42


def link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy(src, dst)


def prepare_split(kit, yolo, split, labels_csv):
    img_src = kit / f"{split}_images"
    img_dst = yolo / "images" / split
    lab_dst = yolo / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True); lab_dst.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(labels_csv)
    df["stem"] = df["image_id"].map(lambda x: Path(str(x)).stem)
    groups = {k: g for k, g in df.groupby("stem")}
    n_boxes = n_skipped = 0
    sizes = {}
    for f in tqdm(sorted(os.listdir(img_src)), desc=f"prepare {split}"):
        stem = Path(f).stem
        link_or_copy(img_src / f, img_dst / f)
        with Image.open(img_src / f) as im:
            W, H = im.size
        sizes[stem] = (W, H)
        lines = []
        for r in (groups[stem].itertuples() if stem in groups else []):
            x1 = max(0.0, min(float(r.x_min), W)); x2 = max(0.0, min(float(r.x_max), W))
            y1 = max(0.0, min(float(r.y_min), H)); y2 = max(0.0, min(float(r.y_max), H))
            if x2 - x1 < 1 or y2 - y1 < 1:
                n_skipped += 1; continue
            lines.append(f"{int(r.class_id)} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
            n_boxes += 1
        (lab_dst / f"{stem}.txt").write_text("\n".join(lines))
    print(f"{split}: {len(sizes)} images, {n_boxes} boxes written, {n_skipped} degenerate skipped")
    return sizes


def degrade(img, rng):
    """1-2 random degradations: blur / low-light / gaussian noise / heavy JPEG."""
    ops = rng.choice(["blur", "dark", "noise", "jpeg"], size=rng.integers(1, 3), replace=False)
    jpeg_q = 95
    for op in ops:
        if op == "blur":
            k = int(rng.choice([3, 5, 7, 9, 11])); img = cv2.GaussianBlur(img, (k, k), 0)
        elif op == "dark":
            f = rng.uniform(0.25, 0.7); gamma = rng.uniform(1.0, 1.8)
            img = (255 * ((img / 255.0) ** gamma) * f).clip(0, 255).astype(np.uint8)
        elif op == "noise":
            sigma = rng.uniform(6, 28)
            img = (img.astype(np.float32) + rng.normal(0, sigma, img.shape)).clip(0, 255).astype(np.uint8)
        elif op == "jpeg":
            jpeg_q = int(rng.integers(8, 30))
    return img, jpeg_q, list(ops)


def yaml_text(path, train, val):
    return f"path: {path}\ntrain: {train}\nval: {val}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, help="competition folder containing train_labels.csv etc.")
    ap.add_argument("--work", default="work")
    a = ap.parse_args()
    kit, work = Path(a.kit), Path(a.work).resolve()
    yolo, yolo_deg = work / "yolo", work / "yolo_degraded"
    work.mkdir(parents=True, exist_ok=True)

    prepare_split(kit, yolo, "train", kit / "train_labels.csv")
    sizes_val = prepare_split(kit, yolo, "val", kit / "val_labels.csv")
    (work / "road.yaml").write_text(yaml_text(yolo, "images/train", "images/val"))

    # degraded copy of val (robustness check)
    rng = np.random.default_rng(SEED)
    src_img, src_lab = yolo / "images" / "val", yolo / "labels" / "val"
    (yolo_deg / "images" / "val").mkdir(parents=True, exist_ok=True)
    if not (yolo_deg / "labels" / "val").exists():
        shutil.copytree(src_lab, yolo_deg / "labels" / "val")
    log = {}
    for f in tqdm(sorted(os.listdir(src_img)), desc="degrade val"):
        out, q, ops = degrade(cv2.imread(str(src_img / f)), rng)
        cv2.imwrite(str(yolo_deg / "images" / "val" / f), out, [cv2.IMWRITE_JPEG_QUALITY, q])
        log[f] = {"ops": ops, "jpeg_q": q}
    (work / "degraded_val_log.json").write_text(json.dumps(log))
    (work / "road_degraded.yaml").write_text(yaml_text(yolo_deg, "images/val", "images/val"))

    # val subsets by image size (600x600, 720x720, everything else)
    groups = {"600": [], "720": [], "other": []}
    for stem, (W, H) in sizes_val.items():
        key = str(W) if (W == H and str(W) in ("600", "720")) else "other"
        groups[key].append(str(yolo / "images" / "val" / f"{stem}.jpg"))
    for k, paths in groups.items():
        if paths:
            lst = work / f"val_{k}.txt"; lst.write_text("\n".join(paths))
            (work / f"road_val_{k}.yaml").write_text(yaml_text(yolo, "images/train", lst))
            print(k, len(paths), "images")


if __name__ == "__main__":
    main()
