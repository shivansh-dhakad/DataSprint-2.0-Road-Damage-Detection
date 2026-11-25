#!/usr/bin/env python3
"""
Local scorer (approximation of the leaderboard formula, written from the competition text).
Per-class 101-point AP at IoU 0.50 and 0.50:0.95; precision/recall at IoU 0.5 for the single confidence threshold
that maximises the mean-over-classes F1; max 100 boxes / image;
accuracy = 100 * (0.30*mAP50 + 0.25*mAP50-95 + 0.15*P + 0.15*R) / 0.85

Use it on a *validation* prediction file made by inference.py:
  python inference.py --source <kit>/val_images --image-ids <kit>/val_image_ids.csv --out val_sub.csv
  python src/evaluate.py --pred val_sub.csv --gt <kit>/val_labels.csv
The organisers' official score.py / check_submission.py are the reference; this is only a convenience copy.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _iou_vec(b, G):
    ix1 = np.maximum(b[0], G[:, 0]); iy1 = np.maximum(b[1], G[:, 1])
    ix2 = np.minimum(b[2], G[:, 2]); iy2 = np.minimum(b[3], G[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ua = (b[2] - b[0]) * (b[3] - b[1]) + (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def _ap101(tp, n_gt):
    if n_gt == 0: return None
    if len(tp) == 0: return 0.0
    tp = np.asarray(tp, float); fp = 1 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt; prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(rec, r, side="left")
        ap += prec[idx] if idx < len(prec) else 0.0
    return ap / 101


def local_score(preds, gt_df, n_classes=4):
    """preds: {stem: [(cls, score, x1, y1, x2, y2)]}; gt_df: image_id,class_id,x_min,y_min,x_max,y_max"""
    thr_list = np.round(np.arange(0.5, 0.96, 0.05), 2)
    gt = {}
    for r in gt_df.itertuples():
        gt.setdefault((Path(str(r.image_id)).stem, int(r.class_id)), []).append([r.x_min, r.y_min, r.x_max, r.y_max])
    gt = {k: np.array(v, float) for k, v in gt.items()}
    n_gt = {c: sum(len(v) for (i, cc), v in gt.items() if cc == c) for c in range(n_classes)}

    ap_by_thr = {c: [] for c in range(n_classes)}
    pr = {}
    for c in range(n_classes):
        items = []
        for img, rows in preds.items():
            for (cc, s, x1, y1, x2, y2) in sorted(rows, key=lambda x: -x[1])[:100]:
                if cc == c: items.append((s, img, np.array([x1, y1, x2, y2])))
        items.sort(key=lambda x: -x[0])
        ious = [(_iou_vec(b, gt[(img, c)]) if (img, c) in gt else None) for _, img, b in items]
        scores = np.array([s for s, _, _ in items])
        for t in thr_list:
            used = {k: np.zeros(len(v), bool) for k, v in gt.items() if k[1] == c}
            tp = []
            for (s, img, b), iv in zip(items, ious):
                hit = 0
                if iv is not None:
                    m = np.where(used[(img, c)], -1, iv)
                    j = int(m.argmax())
                    if m[j] >= t:
                        used[(img, c)][j] = True; hit = 1
                tp.append(hit)
            ap_by_thr[c].append(_ap101(tp, n_gt[c]))
            if t == 0.5: pr[c] = (scores, np.array(tp))
    valid = [c for c in range(n_classes) if n_gt[c] > 0]
    ap50 = float(np.mean([ap_by_thr[c][0] for c in valid]))
    ap5095 = float(np.mean([np.mean(ap_by_thr[c]) for c in valid]))

    taus = np.unique(np.concatenate([pr[c][0] for c in valid])) if any(len(pr[c][0]) for c in valid) else np.array([0.5])
    Ps, Rs, Fs = [], [], []
    for c in valid:
        s, tp = pr[c]; ctp = np.concatenate([[0], np.cumsum(tp)])
        k = np.searchsorted(-s, -taus, side="right")
        TP = ctp[k]; FP = k - TP
        P = TP / np.maximum(TP + FP, 1e-9); R = TP / n_gt[c]
        Ps.append(P); Rs.append(R); Fs.append(2 * P * R / np.maximum(P + R, 1e-9))
    j = int(np.mean(Fs, axis=0).argmax())
    P, R, tau = float(np.mean(Ps, axis=0)[j]), float(np.mean(Rs, axis=0)[j]), float(taus[j])
    acc = 100 * (0.30 * ap50 + 0.25 * ap5095 + 0.15 * P + 0.15 * R) / 0.85
    return dict(mAP50=ap50, mAP50_95=ap5095, P=P, R=R, conf_thr=tau, accuracy_score=float(acc))


def read_submission(path):
    """submission csv (image_id,PredictionString) -> {stem: [(cls, score, x1, y1, x2, y2)]}"""
    df = pd.read_csv(path)
    out = {}
    for iid, ps in zip(df["image_id"], df["PredictionString"].fillna("none")):
        v = str(ps).split()
        out[Path(str(iid)).stem] = [] if (not v or v[0] == "none") else \
            [(int(float(v[i])), float(v[i + 1]), *map(float, v[i + 2:i + 6])) for i in range(0, len(v), 6)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="submission-format csv produced for the VAL images")
    ap.add_argument("--gt", required=True, help="val_labels.csv")
    a = ap.parse_args()
    s = local_score(read_submission(a.pred), pd.read_csv(a.gt))
    print({k: round(v, 4) for k, v in s.items()})


if __name__ == "__main__":
    main()
