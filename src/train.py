#!/usr/bin/env python3
"""
Step 2 - train a YOLO26s detector (the two models used in the final ensemble).

  # main model  (inferred later at 1088 px with TTA)          ~11 h on a Kaggle GPU
  python src/train.py --data work/road.yaml --imgsz 1024 --batch 32 --close-mosaic 8  --warmup 2 --name yolo26s_1024

  # second model (640 px)                                      ~4 h
  python src/train.py --data work/road.yaml --imgsz 640  --batch 32 --close-mosaic 10 --warmup 3 --name yolo26s_640

Same hyper-parameters as logged in results/<run>/args.yaml. Weights land in runs/<name>/weights/best.pt.
"""
import argparse
import os
from pathlib import Path


def build_albu():
    """Extra degradations for robustness (blur, noise, JPEG, low light). Albumentations v1/v2 arg names differ -> try both."""
    try:
        import albumentations as A
    except Exception:
        return None
    for make in (
        lambda: [A.GaussianBlur(blur_limit=(3, 9), p=0.15),
                 A.GaussNoise(std_range=(0.02, 0.1), p=0.15),
                 A.ImageCompression(quality_range=(10, 40), p=0.2),
                 A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.15), contrast_limit=0.3, p=0.3),
                 A.RandomGamma(gamma_limit=(80, 200), p=0.15)],
        lambda: [A.GaussianBlur(blur_limit=(3, 9), p=0.15),
                 A.GaussNoise(var_limit=(10, 600), p=0.15),
                 A.ImageCompression(quality_lower=10, quality_upper=40, p=0.2),
                 A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.15), contrast_limit=0.3, p=0.3),
                 A.RandomGamma(gamma_limit=(80, 200), p=0.15)]):
        try:
            return make()
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="road.yaml written by prepare_data.py")
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--close-mosaic", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=2)
    ap.add_argument("--name", default=None)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # fewer OOMs at 1024 px
    from ultralytics import YOLO

    name = a.name or f"{Path(a.model).stem}_{a.imgsz}"
    kw = dict(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device,
        seed=a.seed, deterministic=True, workers=4, patience=a.patience, cos_lr=True,
        close_mosaic=a.close_mosaic, warmup_epochs=a.warmup,
        # geometry: roads have a fixed vertical orientation -> no vertical flip / rotation
        fliplr=0.5, flipud=0.0, degrees=0.0, translate=0.1, scale=0.4, mosaic=1.0, mixup=0.0,
        # colour: strong value jitter for low-light images
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,
        project=a.project, name=name, exist_ok=True, plots=True,
    )
    albu = build_albu()
    if albu:
        kw["augmentations"] = albu
    model = YOLO(a.model)
    try:
        model.train(**kw)
    except TypeError:
        kw.pop("augmentations", None)
        model.train(**kw)
    # resume an interrupted run:  YOLO("runs/<name>/weights/last.pt").train(resume=True)


if __name__ == "__main__":
    main()
