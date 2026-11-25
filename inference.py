"""
inference.py - DataSprint 2.0 Road Damage Detection (follows the organisers' contract).

The benchmark imports this file from the repo root and calls:
    model = load(device)                 # once, not timed; device "cuda" or "cpu"
    detections = predict(model, image)   # per image, batch size 1
      image      : numpy [H, W, 3] uint8, RGB, original size
      detections : list of (class_id, score, x_min, y_min, x_max, y_max) in original pixels, score >= 0.001, <= 100

Final pipeline (default preset "final"):
    YOLO26s trained @1024 -> run @1088 with TTA, NMS IoU 0.45
  + YOLO26s trained @640  -> run @640, NMS IoU 0.45
  -> Weighted Box Fusion (IoU 0.55, weights 1:1), top 100 boxes.
Weights are loaded from ./weights next to this file; no internet is used.

Command line (builds a Kaggle submission CSV, same code path as the benchmark):
    python inference.py --source <test_images> --image-ids <test_image_ids.csv> --out submission.csv
    python inference.py --source <folder> --out demo.csv --preset single        # fastest
"""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_WEIGHTS = os.path.join(_HERE, "weights", "yolo26s_1024_best.pt")     # trained @1024
SECOND_WEIGHTS = os.path.join(_HERE, "weights", "yolo26s_640_best.pt")    # trained @640

PRESETS = {
    "final": dict(imgsz=1088, iou=0.45, tta=True, ensemble=True),    # = submission.csv (val 49.39 local)
    "v1": dict(imgsz=1024, iou=0.50, tta=False, ensemble=True),      # = submissions/submission_v1 (val 48.42 local)
    "single": dict(imgsz=1024, iou=0.50, tta=False, ensemble=False), # fastest
}
SECOND_IMGSZ = 640
WBF_IOU = 0.55
WBF_WEIGHTS = [1.0, 1.0]
CONF = 0.001
MAX_DET = 100
N_CLASSES = 4


# ----------------------------------------------------------------------------- fusion
def iou_vec(b, G):
    ix1 = np.maximum(b[0], G[:, 0]); iy1 = np.maximum(b[1], G[:, 1])
    ix2 = np.minimum(b[2], G[:, 2]); iy2 = np.minimum(b[3], G[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ua = (b[2] - b[0]) * (b[3] - b[1]) + (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def wbf_image(rows_per_model, weights=None, iou_thr=WBF_IOU, max_det=MAX_DET):
    """Weighted Box Fusion for ONE image. rows_per_model: list (one per model) of [(cls, score, x1,y1,x2,y2)].
    Per class, boxes (score*model weight) are clustered greedily by IoU with the running fused box; the fused box is the
    score-weighted mean of its members, its score = mean member score * (#members / #models)."""
    n_models = len(rows_per_model)
    weights = weights or [1.0] * n_models
    fused = []
    for c in range(N_CLASSES):
        cand = [(s * weights[m], np.array([x1, y1, x2, y2], float))
                for m, rows in enumerate(rows_per_model) for (cc, s, x1, y1, x2, y2) in rows if cc == c]
        cand.sort(key=lambda x: -x[0])
        clusters = []
        for s, b in cand:
            best = None
            if clusters:
                ious = iou_vec(b, np.array([cl["fb"] for cl in clusters]))
                j = int(ious.argmax())
                if ious[j] > iou_thr:
                    best = j
            if best is None:
                clusters.append(dict(boxes=[b], scores=[s], fb=b.copy()))
            else:
                cl = clusters[best]
                cl["boxes"].append(b); cl["scores"].append(s)
                w = np.array(cl["scores"])
                cl["fb"] = (np.array(cl["boxes"]) * w[:, None]).sum(0) / w.sum()
        for cl in clusters:
            conf = float(np.mean(cl["scores"])) * min(len(cl["scores"]), n_models) / n_models
            fused.append((c, conf, *map(float, cl["fb"])))
    return sorted(fused, key=lambda x: -x[1])[:max_det]


# ----------------------------------------------------------------------------- contract
class _Model:
    def __init__(self, main, second, device, cfg):
        self.main, self.second, self.device, self.cfg = main, second, device, cfg
        self.end2end_kw = {"end2end": False}   # YOLO26: classic NMS (what was tuned); dropped if unsupported


def load(device="cuda", preset="final"):
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    import logging
    LOGGER.setLevel(logging.ERROR)
    cfg = dict(PRESETS[preset])
    dev = 0 if str(device).startswith("cuda") else "cpu"
    main = YOLO(MAIN_WEIGHTS)
    second = YOLO(SECOND_WEIGHTS) if cfg["ensemble"] else None
    return _Model(main, second, dev, cfg)


def _run(model, yolo, bgr, imgsz, augment):
    kw = dict(imgsz=imgsz, conf=CONF, iou=model.cfg["iou"], max_det=MAX_DET, device=model.device, verbose=False)
    if augment:
        kw["augment"] = True
    try:
        r = yolo.predict(bgr, **kw, **model.end2end_kw)[0]
    except (TypeError, SyntaxError, KeyError, ValueError):
        model.end2end_kw = {}                  # older ultralytics without the flag
        r = yolo.predict(bgr, **kw)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy(); conf = r.boxes.conf.cpu().numpy(); cls = r.boxes.cls.cpu().numpy().astype(int)
    return [(int(c), float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3])) for c, s, b in zip(cls, conf, xyxy)]


def predict(model, image):
    bgr = np.ascontiguousarray(image[:, :, ::-1])     # Ultralytics treats numpy arrays as BGR
    cfg = model.cfg
    rows = _run(model, model.main, bgr, cfg["imgsz"], cfg["tta"])
    if cfg["ensemble"]:
        rows_b = _run(model, model.second, bgr, SECOND_IMGSZ, False)
        rows = wbf_image([rows, rows_b], WBF_WEIGHTS, WBF_IOU, MAX_DET)
    return sorted(rows, key=lambda x: -x[1])[:MAX_DET]


# ----------------------------------------------------------------------------- CLI: images -> submission csv
def _pred_string(rows):
    if not rows:
        return "none"
    return " ".join(f"{c} {s:.6f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}" for c, s, x1, y1, x2, y2 in rows)


def main():
    import argparse
    from pathlib import Path
    import cv2
    import pandas as pd
    from tqdm.auto import tqdm

    ap = argparse.ArgumentParser(description="Road damage detection -> submission csv")
    ap.add_argument("--source", required=True, help="folder with images")
    ap.add_argument("--image-ids", default=None, help="csv with image_id column (fixes row order); default: all images")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--detailed-out", default=None, help="optional per-box csv")
    ap.add_argument("--preset", choices=list(PRESETS), default="final")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = {p.stem: p for p in sorted(Path(a.source).iterdir()) if p.suffix.lower() in ext}
    if not files:
        raise SystemExit(f"no images in {a.source}")
    if a.image_ids:
        df = pd.read_csv(a.image_ids)
        ids = df["image_id" if "image_id" in df.columns else df.columns[0]].astype(str).tolist()
        miss = [i for i in ids if Path(i).stem not in files]
        if miss:
            raise SystemExit(f"{len(miss)} ids have no image in {a.source} (e.g. {miss[:3]})")
    else:
        ids = sorted(files)
    import torch
    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    model = load(device, a.preset)
    print("preset:", a.preset, model.cfg, "| device:", device)

    out, detailed = [], []
    for i in tqdm(ids):
        img = cv2.cvtColor(cv2.imread(str(files[Path(i).stem])), cv2.COLOR_BGR2RGB)
        rows = predict(model, img)
        out.append(_pred_string(rows))
        detailed += [(i, c, round(s, 5), round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)) for c, s, x1, y1, x2, y2 in rows]
    pd.DataFrame({"image_id": ids, "PredictionString": out}).to_csv(a.out, index=False)
    print(f"saved {a.out} | rows {len(ids)} | boxes {len(detailed)} | 'none': {out.count('none')}")
    if a.detailed_out:
        pd.DataFrame(detailed, columns=["image_id", "class_id", "score", "x_min", "y_min", "x_max", "y_max"]).to_csv(a.detailed_out, index=False)


if __name__ == "__main__":
    main()