# DataSprint 2.0 – Road Damage Detection

Detect and classify road damage in street-level photos (RDD2020: India, Japan, Czech Republic) with an **ensemble of two YOLO26s detectors** (1024 px + 640 px), test-time augmentation and Weighted Box Fusion.

| Class id | Code | Damage type |
|---|---|---|
| 0 | D00 | Longitudinal crack |
| 1 | D10 | Transverse crack |
| 2 | D20 | Alligator crack |
| 3 | D40 | Pothole |

> **Want to see the detections instead of reading a CSV?** Test the model visually with the demo repository: **https://github.com/shivansh-dhakad/PAVEXA**

Author: Shivansh Dhakad ([@shivansh-dhakad](https://github.com/shivansh-dhakad)) · Competition: DataSprint 2.0 (Anveshan, NIE Mysuru), Kaggle, 23–25 Nov 2026

---

## Table of contents
1. [Project overview](#1-project-overview)
2. [Problem statement & evaluation](#2-problem-statement--evaluation)
3. [Dataset & exploratory analysis](#3-dataset--exploratory-analysis)
4. [Approach](#4-approach)
5. [Model architecture](#5-model-architecture)
6. [Training details](#6-training-details)
7. [Inference pipeline (TTA + ensemble)](#7-inference-pipeline-tta--ensemble)
8. [Results](#8-results)
9. [Speed](#9-speed)
10. [Robustness](#10-robustness)
11. [Repository structure & how to run](#11-repository-structure--how-to-run)
12. [Limitations, future work & credits](#12-limitations-future-work--credits)

---

## 1. Project overview
The pipeline turns an image into a list of boxes `(class, confidence, x_min, y_min, x_max, y_max)`:

1. Two YOLO26s models, fine-tuned from COCO weights on the competition training set, are run on every image — the main model at **1088 px with test-time augmentation (TTA)**, the second at **640 px**.
2. Their predictions are merged with **Weighted Box Fusion (WBF)**.
3. At most 100 boxes per image are written to `submission.csv` (`image_id,PredictionString`).

**Quick start**
```bash
git clone https://github.com/shivansh-dhakad/DataSprint-2.0-Road-Damage-Detection
cd DataSprint-2.0-Road-Damage-Detection
pip install -r requirements.txt
python inference.py --source <folder_with_images> --out predictions.csv      # works on any folder of road photos
```

## 2. Problem statement & evaluation
Object detection with four damage classes. Per the competition description, the final score combines accuracy (mAP@50, mAP@50:95, precision, recall), inference speed on a T4 GPU and robustness to degraded images. Our local scorer (`src/evaluate.py`) follows the accuracy part of that text:

`accuracy = 100 · (0.30·mAP50 + 0.25·mAP50-95 + 0.15·P + 0.15·R) / 0.85`

with per-class 101-point AP, and P/R at IoU 0.5 for the single confidence threshold that maximises mean-over-classes F1. The organisers' `score.py`, `check_submission.py` and `benchmark_speed.py` (from the competition kit, not redistributed here) are the reference.

Submission format: one row per test image, `PredictionString = "cls score x1 y1 x2 y2 cls score ..."` (pixels, original image size), `none` when there is no detection, max 100 boxes per image.

## 3. Dataset & exploratory analysis
Competition kit (RDD2020 subset). Full EDA is in [`notebooks/01_understanding_the_data.ipynb`](notebooks/01_understanding_the_data.ipynb).

| Split | Images | Boxes |
|---|---|---|
| train | 14,691 | 17,517 |
| val | 2,150 | 2,561 |
| test | 4,200 | – (hidden) |

* **Class balance (train):** D00 4,600 (26.3 %) · D10 3,132 (17.9 %) · D20 5,864 (33.5 %) · D40 3,921 (22.4 %). Val has the same proportions.
* **Background images:** 42.0 % of train and 41.7 % of val images contain no damage (1.19 boxes/image on average, max 14) → the model must stay quiet on clean roads.
* **Image sizes (train):** 600×600 = 9,123 · 720×720 = 5,356 · 1024×1024 = 98 · 540×540 = 84 · 1080×1080 = 30. Sizes roughly map to country/camera and behave very differently (see §8).
* **Box shapes:** D10 are wide and thin (median 0.215 × 0.042 of the image), D00 are taller, D20 are the largest (median 0.345 × 0.243), D40 potholes are small (0.103 × 0.060) → high input resolution matters.
* **Label checks:** no nulls, no degenerate/negative boxes, no duplicate rows, no missing images.

## 4. Approach
1. `src/prepare_data.py` converts the CSV labels to YOLO format (background images get empty label files) and builds a **degraded copy of val** (blur / low light / noise / heavy JPEG) plus val subsets per image size, for robustness checks.
2. `src/train.py` fine-tunes `yolo26s.pt` at **1024 px** (main model) and at **640 px** (second model) with road-specific augmentation.
3. Inference-time choices were tuned on val (NMS IoU, input size, box cap, per-class score calibration, TTA, ensemble). Per-class calibration and a lower box cap gave no gain and are **not** used; TTA and the 2-model ensemble did and are.
4. `inference.py` runs the final configuration end-to-end.

## 5. Model architecture
* **Detector:** Ultralytics **YOLO26s** (small variant) – anchor-free single-stage detector, initialised from the COCO-pretrained `yolo26s.pt`, 4 output classes. Checkpoint size ≈ 20 MB each.
* **NMS:** YOLO26 has an NMS-free (end-to-end) head. We switch it off (`end2end=False`) and use classic NMS (IoU 0.45) at inference because that is the setting we tuned and validated.
* **Ensemble members:** `yolo26s_1024` (trained at 1024, run at 1088) and `yolo26s_640` (trained and run at 640). A third model, YOLOv8s @640, was trained as a baseline only and is **not** part of the submission (training log in `results/yolov8s_640_baseline/`, no weights shipped).

## 6. Training details
Logged in `results/<run>/args.yaml` and `results/<run>/results.csv`.

| Setting | yolo26s_1024 (main) | yolo26s_640 (second) |
|---|---|---|
| Image size | 1024 | 640 |
| Epochs / patience | 60 / 15 (ran all 60) | 60 / 15 (ran all 60) |
| Batch | 32 | 32 |
| Warm-up epochs / close_mosaic | 2 / 8 | 3 / 10 |
| LR schedule | cosine, lr0 0.01, lrf 0.01 | same |
| Momentum / weight decay | 0.937 / 5e-4 | same |
| Seed / deterministic | 42 / True | 42 / True |
| Precision | AMP | AMP |
| Wall-clock (Kaggle GPU) | ≈ 11.0 h | ≈ 4.2 h |

**Augmentation** (roads have a fixed vertical orientation → no vertical flip, no rotation):
mosaic 1.0 · horizontal flip 0.5 · translate 0.1 · scale 0.4 · HSV (h 0.015, s 0.6, v 0.5 for low light) · random erasing 0.4 · RandAugment · plus Albumentations for robustness: Gaussian blur (p 0.15), Gaussian noise (p 0.15), JPEG compression q 10–40 (p 0.2), brightness/contrast (p 0.3), random gamma (p 0.15).

## 7. Inference pipeline (TTA + ensemble)
Final configuration (`--preset final`, the default of `inference.py`):

| Step | Setting |
|---|---|
| Main model | `weights/yolo26s_1024_best.pt`, **imgsz 1088**, **TTA on** (`augment=True`), NMS IoU 0.45, conf ≥ 0.001 |
| Second model | `weights/yolo26s_640_best.pt`, imgsz 640, NMS IoU 0.45, conf ≥ 0.001 |
| Fusion | Weighted Box Fusion per class, cluster IoU 0.55, model weights 1 : 1 |
| WBF score | mean score of cluster × (#models that found it / 2) → boxes seen by only one model are halved |
| Output | top-100 boxes per image by score, coordinates in original pixels, 6 decimals for score |

conf ≥ 0.001 keeps nearly every candidate: mAP rewards a complete ranking, and precision/recall are measured at the best single threshold afterwards. WBF is our own implementation (`wbf_image()` in `inference.py`), no extra library.

**Organisers' interface.** `inference.py` sits in the repo root and follows the competition contract: `load(device)` is called once (loads both checkpoints from `./weights`, no internet) and `predict(model, image)` is called per image (RGB `uint8` array, original size) and returns `(class_id, score, x_min, y_min, x_max, y_max)` tuples in original pixels, at most 100, scores ≥ 0.001. TTA and the ensemble run inside `predict()`, so their time is counted by `benchmark_speed.py`. `load(device, preset=...)` accepts `final` (default), `v1` or `single`.

## 8. Results
All numbers are on the 2,150 **validation** images (the test labels are hidden). "Local" = our scorer in `src/evaluate.py`, "official" = the organisers' `score.py`.

| Configuration | Accuracy score | mAP@50 | mAP@50:95 | P | R |
|---|---|---|---|---|---|
| YOLO26s @640 (baseline, official) | 44.50 | 0.5263 | 0.2270 | 0.5841 | 0.5065 |
| YOLO26s @1024 (official) | 46.19 | 0.5472 | 0.2380 | 0.6054 | 0.5209 |
| @1088, NMS 0.45 (local) | 45.90 | – | – | – | – |
| @1088 + TTA (local) | 47.95 | 0.5709 | 0.2519 | – | – |
| @1088 + 640 model ensemble, no TTA (local) | 48.54 | – | – | – | – |
| **v3: @1088 + TTA + 640 ensemble (local)** | **49.39** | **0.5910** | **0.2608** | **0.6460** | **0.5362** |

Sweep for the **v1** settings (1024 px, NMS IoU 0.5, local scorer): single model 46.51 · + TTA 47.16 (mAP50 0.5622, mAP50-95 0.2509) · + 640 ensemble, WBF weights 1:1 → **48.42** (weights 2:1 → 48.48).

The inference settings were selected on this same val set, so the final number is slightly optimistic. Per-class AP@50 of the single 1024 model (Ultralytics val, IoU 0.6): D00 0.506 · D10 0.523 · D20 0.710 · D40 0.546 – alligator cracks are easiest, thin transverse and longitudinal cracks hardest.

**Submission files** (test set, 4,200 rows, passes the kit's `check_submission.py`):

| File | Description | Boxes | Images with `none` |
|---|---|---|---|
| `submission.csv` (= `submissions/submission_v3_ensemble_tta_1088_640.csv`) | **v3** – 1088 px + TTA + 640 ensemble, NMS IoU 0.45 (val 49.39) | 101,761 | 163 |
| `submissions/submission_v1_ensemble_1024_640.csv` | **v1** – 1024 px + 640 ensemble, NMS IoU 0.5, no TTA (val 48.42) | 115,727 | 158 |

Both were submitted. `inference.py --preset final` reproduces v3, `--preset v1` reproduces v1.

Training curves, PR/F1 curves, confusion matrices and sample predictions are in `results/`.

## 9. Speed
Measured in the notebook, one image at a time, wall-clock including image loading (`notebooks/03_…`):

| Configuration | ms / image |
|---|---|
| Main model @1088 | 41.5 |
| Main model @1088 + TTA | 93.7 |
| Main @1088 + 640 ensemble (no TTA) | 60.7 |
| Final v3 (TTA + ensemble) | not timed separately; roughly 93.7 + ~19 ≈ 110 ms |
| v1 settings: 1024 px, no TTA | 37.5 |
| v1 settings: 1024 px + TTA | 83.1 |
| **v1: 1024 px + 640 ensemble (no TTA)** | **56.7** |

TTA and the ensemble buy about +3.5 accuracy points at 2–3× the latency. Use `--preset single` (1 model, no TTA) when speed matters more than the last accuracy points. The organisers' `benchmark_speed.py` is the reference for the speed score.

## 10. Robustness
* Training uses blur, noise, JPEG and low-light augmentation (§6).
* A **degraded val set** (random blur / darkening / noise / JPEG 8–30) is built by `src/prepare_data.py`. Ultralytics val on the 1024 model: mAP50 0.571 → 0.404 and mAP50-95 0.255 → 0.173 under degradation; the 640 model drops 0.546 → 0.413 (`results/eval_summary_*.csv`). The 640 model degrades slightly less, which is one reason the ensemble helps.
* **By image size** (1024 model, mAP50): 600×600 → 0.618, 720×720 → 0.417. The 720 px group is clearly harder; this is the main remaining weakness.

## 11. Repository structure & how to run
```
.
├── README.md
├── requirements.txt
├── inference.py              # final pipeline: 2 models + TTA + WBF -> submission CSV
├── submission.csv            # FINAL submission (v3)
├── submissions/              # both submitted CSVs (v1 no-TTA ensemble, v3 final)
├── weights/
│   ├── yolo26s_1024_best.pt  # main model
│   └── yolo26s_640_best.pt   # second model
├── src/
│   ├── prepare_data.py       # CSV -> YOLO labels, degraded val, size subsets
│   ├── train.py              # training (both models)
│   └── evaluate.py           # local scorer
├── notebooks/
│   ├── 01_understanding_the_data.ipynb
│   ├── 02_baseline_yolo26s_640.ipynb
│   └── 03_train_1024_and_ensemble_inference.ipynb   # 1024 training + inference sweeps + final ensemble
├── configs/                  # dataset yaml files used on Kaggle (paths are Kaggle paths)
└── results/                  # logs, curves, confusion matrices, eval summaries for each run
```

**Requirements:** Python 3.10+, a CUDA GPU recommended (CPU works but is slow), `pip install -r requirements.txt` (YOLO26 needs a recent Ultralytics release).

**Run inference / reproduce the submission**
```bash
KIT=/path/to/competition_data
python inference.py --source $KIT/test_images --image-ids $KIT/test_image_ids.csv --out submission.csv
python $KIT/check_submission.py --pred submission.csv --image-ids $KIT/test_image_ids.csv   # organisers' checker
```
The command line calls the same `load()` / `predict()` as the benchmark (one image at a time). Options: `--preset {final,v1,single}`, `--device {cuda,cpu}`, `--detailed-out boxes.csv`.

**Speed benchmark (organisers' script)**
```bash
python $KIT/benchmark_speed.py   # imports ./inference.py: load(device) then predict(model, image)
```
Regenerated CSVs can differ marginally from the shipped one (GPU non-determinism, Ultralytics version).

**Check accuracy on validation**
```bash
python inference.py --source $KIT/val_images --image-ids $KIT/val_image_ids.csv --out val_sub.csv
python src/evaluate.py --pred val_sub.csv --gt $KIT/val_labels.csv
```

**Retrain from scratch**
```bash
python src/prepare_data.py --kit $KIT --work work
python src/train.py --data work/road.yaml --imgsz 1024 --batch 32 --close-mosaic 8  --warmup 2 --name yolo26s_1024
python src/train.py --data work/road.yaml --imgsz 640  --batch 32 --close-mosaic 10 --warmup 3 --name yolo26s_640
# copy runs/<name>/weights/best.pt to weights/yolo26s_1024_best.pt and weights/yolo26s_640_best.pt
```

**Test the model visually:** open **https://github.com/shivansh-dhakad/PAVEXA** to try the detector on images and see the boxes drawn.

## 12. Limitations, future work & credits
* Thin cracks (D00, D10) and the 720×720 image group remain the weak spots; severe blur/darkness lowers recall.
* The final config is slow-ish (two models + TTA); a distilled single model or TensorRT/FP16 export would cut latency.
* Possible next steps: larger backbone (m/l), longer training, class-balanced sampling, a third ensemble member, tiling for high-resolution images.
* Data: RDD2020 (Arya et al., Road Damage Detection and Classification, IEEE Big Data Cup 2020) as provided by the organisers. Framework: [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0 – respect its license when redistributing).