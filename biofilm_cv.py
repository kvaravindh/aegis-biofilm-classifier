#!/usr/bin/env python3
"""
biofilm_cv.py — tell B. subtilis (NCIB3610) and E. coli (MG1655) biofilm apart in
crystal-violet brightfield images of a co-inoculated well.

How it works
------------
1. Stain map: the image is converted to optical density (OD) of the green channel,
   where crystal violet absorbs most strongly. Colour/hue is deliberately thrown away,
   because between photos it mostly reflects white balance, not species.
2. Biomass mask: pixels with OD above an Otsu threshold are "biofilm"; the rest is
   unstained background and is excluded from all percentages.
3. Morphology features: the OD map is locally contrast-normalised (so absolute stain
   density can't be used as a shortcut) and described per pixel at several scales —
   smoothed intensity, edge strength, Hessian ridge/blob responses (chains vs short
   rods), structure-tensor coherence (aligned chains / bundles) and local variance.
4. Pixel classifier: a 60-tree random forest trained on pixels from monoculture reference
   images (every biomass pixel in the B. subtilis image = B. subtilis, etc.).
5. Co-culture call: each biomass pixel is classified, then the local B. subtilis
   fraction is computed in a window (default radius 15 px). A pixel is
     - B. subtilis  if local B. subtilis fraction >= 1 - overlap_thr
     - E. coli      if local B. subtilis fraction <= overlap_thr
     - Overlap      otherwise (both species present within the window = intermixed)
   Reported: % of biomass area that is B. subtilis-only, E. coli-only and overlap
   (these sum to 100), plus the overall composition (B. sub vs E. coli share, sums to
   100, overlap area split by its local fractions).

CLI
---
  python biofilm_cv.py train   --bsub ref_bsub1.png [ref_bsub2.png ...] --ecoli ref_ecoli1.png [...] --model model.npz
  python biofilm_cv.py predict --model model.npz --out results/ coculture1.png [coculture2.png ...]
  python biofilm_cv.py validate --bsub ... --ecoli ... --out validation/

All images (references and co-cultures) must be taken with the SAME objective, camera
zoom and resolution; otherwise use --scale to resample them to a common pixel size.
"""
from __future__ import annotations

import argparse
import csv
import warnings
import json
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier



# ----------------------------------------------------------------------------- image I/O

import bfcore as core
from bfcore import BSUB, ECOLI, Forest

RF_PARAMS = dict(n_estimators=60, max_depth=14, min_samples_leaf=10)


def load_rgb(path: str, scale: float = 1.0) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if scale != 1.0:
        w, h = im.size
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    return np.asarray(im).astype(np.float32)


stain_od = core.stain_od


def biomass_mask(od):
    return core.biomass_mask(od)


def image_features(od, mask=None):
    return core.features(od)


# ----------------------------------------------------------------------------- model

@dataclass
class ModelMeta:
    scale: float = 1.0
    window: int = 15
    overlap_thr: float = 0.25
    tpr: float = 1.0   # P(called B. sub | truly B. sub), estimated on held-out reference regions
    fpr: float = 0.0   # P(called B. sub | truly E. coli)
    n_bsub_px: int = 0
    n_ecoli_px: int = 0
    bsub_refs: list | None = None
    ecoli_refs: list | None = None


def _sample(feat, mask, n, rng, region=None):
    m = mask if region is None else (mask & region)
    idx = np.flatnonzero(m.ravel())
    if len(idx) > n:
        idx = rng.choice(idx, n, replace=False)
    return feat.reshape(-1, feat.shape[-1])[idx]


def fit_rf(Xb, Xe, seed=0):
    X = np.vstack([Xb, Xe])
    y = np.r_[np.full(len(Xb), BSUB), np.full(len(Xe), ECOLI)]
    rf = RandomForestClassifier(**RF_PARAMS, class_weight="balanced", n_jobs=-1, random_state=seed)
    rf.fit(X, y)
    return Forest.from_sklearn(rf)


def prep(path, scale):
    od = stain_od(load_rgb(path, scale))
    mask, _ = biomass_mask(od)
    return od, mask, image_features(od)


def _split_region(allowed: np.ndarray, part: int) -> np.ndarray:
    """Split an allowed region into two halves (with a gap) along its longer side."""
    ys, xs = np.nonzero(allowed)
    axis = 0 if (ys.max() - ys.min()) >= (xs.max() - xs.min()) else 1
    c = ys if axis == 0 else xs
    cut = (c.min() + c.max()) // 2
    grid = np.indices(allowed.shape)[axis]
    keep = grid < cut - 16 if part == 0 else grid > cut + 16
    return allowed & keep


def fit_calibrated(B, E, region=None, per_image=40000, seed=0):
    """B, E: lists of (od, mask, features). region(shape)->bool restricts training pixels.
    Returns (forest, tpr, fpr, n_b, n_e). tpr/fpr come from 2-fold spatial cross-validation
    inside the allowed region and de-bias composition estimates (adjusted classify-and-count)."""
    rng = np.random.default_rng(seed)
    allowed = lambda m: (np.ones(m.shape, bool) if region is None else region(m.shape))
    tp, fp = [], []
    for part in (0, 1):
        Xb = np.vstack([_sample(F, m, per_image // 2, rng, _split_region(allowed(m), part)) for _, m, F in B])
        Xe = np.vstack([_sample(F, m, per_image // 2, rng, _split_region(allowed(m), part)) for _, m, F in E])
        fo = fit_rf(Xb, Xe, seed)
        for _, m, F in B:
            tp.append(np.mean(fo.predict(F[m & _split_region(allowed(m), 1 - part)]) >= 0.5))
        for _, m, F in E:
            fp.append(np.mean(fo.predict(F[m & _split_region(allowed(m), 1 - part)]) >= 0.5))
    Xb = np.vstack([_sample(F, m, per_image, rng, allowed(m)) for _, m, F in B])
    Xe = np.vstack([_sample(F, m, per_image, rng, allowed(m)) for _, m, F in E])
    return fit_rf(Xb, Xe, seed), float(np.mean(tp)), float(np.mean(fp)), len(Xb), len(Xe)


def train(bsub_paths, ecoli_paths, scale=1.0, per_image=40000, seed=0):
    B = [prep(p, scale) for p in bsub_paths]
    E = [prep(p, scale) for p in ecoli_paths]
    fo, tpr, fpr, nb, ne = fit_calibrated(B, E, None, per_image, seed)
    meta = ModelMeta(scale=scale, tpr=tpr, fpr=fpr, n_bsub_px=nb, n_ecoli_px=ne,
                     bsub_refs=[os.path.basename(p) for p in bsub_paths],
                     ecoli_refs=[os.path.basename(p) for p in ecoli_paths])
    return fo, meta


def save_model(path, fo: Forest, meta: ModelMeta):
    """Plain-array model file (.npz): no pickle, works across scikit-learn versions."""
    np.savez_compressed(path, feat=fo.feat, thr=fo.thr, left=fo.left, right=fo.right, roots=fo.roots,
                        meta=np.array(json.dumps(asdict(meta))))


def load_model(path):
    z = np.load(path, allow_pickle=False)
    fo = Forest(z["feat"], z["thr"], z["left"], z["right"], z["roots"])
    return fo, ModelMeta(**json.loads(str(z["meta"])))


# ----------------------------------------------------------------------------- prediction

def classify(fo, od, mask, window=15, overlap_thr=0.25, tpr=1.0, fpr=0.0):
    return core.classify(fo, od, mask, window, overlap_thr, tpr, fpr)


COLORS = {0: (0, 0, 0), 1: (30, 144, 255), 2: (255, 140, 0), 3: (220, 40, 160)}  # blue, orange, magenta


def overlay(rgb, lab, alpha=0.45):
    out = rgb.copy()
    for k, c in COLORS.items():
        if k == 0:
            continue
        m = lab == k
        out[m] = (1 - alpha) * out[m] + alpha * np.array(c, np.float32)
    return out.clip(0, 255).astype(np.uint8)


def save_figure(path, rgb, res, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    s = res["stats"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6))
    ax[0].imshow(rgb.astype(np.uint8)); ax[0].set_title("Input")
    im = ax[1].imshow(np.where(res["labels"] > 0, res["frac_bsub"], np.nan), cmap="coolwarm_r", vmin=0, vmax=1)
    ax[1].set_title("Local B. subtilis fraction (biomass only)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02)
    ax[2].imshow(overlay(rgb, res["labels"]))
    ax[2].set_title("Call")
    ax[2].legend(handles=[Patch(color=np.array(COLORS[1]) / 255, label=f"B. subtilis only  {s['bsub_only_pct']:.1f}%"),
                          Patch(color=np.array(COLORS[2]) / 255, label=f"E. coli only  {s['ecoli_only_pct']:.1f}%"),
                          Patch(color=np.array(COLORS[3]) / 255, label=f"Overlap  {s['overlap_pct']:.1f}%")],
                 loc="lower right", fontsize=9, framealpha=0.85)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{title}   |   composition: B. subtilis {s['bsub_total_pct']:.1f}%  /  E. coli {s['ecoli_total_pct']:.1f}%"
                 f"   (biomass covers {s['biomass_area_pct_of_image']:.0f}% of field)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def predict_files(model_path, paths, out_dir, window=None, overlap_thr=None, scale=None):
    rf, meta = load_model(model_path)
    window = window or meta.window
    overlap_thr = overlap_thr if overlap_thr is not None else meta.overlap_thr
    scale = scale or meta.scale
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for p in paths:
        rgb = load_rgb(p, scale)
        od = stain_od(rgb)
        mask, _ = biomass_mask(od)
        res = classify(rf, od, mask, window, overlap_thr, meta.tpr, meta.fpr)
        stem = os.path.splitext(os.path.basename(p))[0]
        save_figure(os.path.join(out_dir, f"{stem}_result.png"), rgb, res, stem)
        Image.fromarray(overlay(rgb, res["labels"])).save(os.path.join(out_dir, f"{stem}_overlay.png"))
        Image.fromarray(res["labels"]).save(os.path.join(out_dir, f"{stem}_labels.png"))  # 0 bg,1 Bs,2 Ec,3 overlap
        row = {"image": os.path.basename(p), **{k: round(v, 2) for k, v in res["stats"].items()}}
        rows.append(row)
        s = res["stats"]
        print(f"{row['image']}: B. subtilis only {s['bsub_only_pct']:.1f}% | E. coli only {s['ecoli_only_pct']:.1f}% | "
              f"overlap {s['overlap_pct']:.1f}%   (composition B.sub {s['bsub_total_pct']:.1f}% / E.coli {s['ecoli_total_pct']:.1f}%)")
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


# ----------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train on monoculture reference images")
    t.add_argument("--bsub", nargs="+", required=True)
    t.add_argument("--ecoli", nargs="+", required=True)
    t.add_argument("--model", default="model/biofilm_model.npz")
    t.add_argument("--scale", type=float, default=1.0, help="resample factor applied to every image")
    t.add_argument("--window", type=int, default=15, help="overlap window radius in px (after scaling)")
    t.add_argument("--overlap-thr", type=float, default=0.25)

    p = sub.add_parser("predict", help="score co-culture images")
    p.add_argument("images", nargs="+")
    p.add_argument("--model", default="model/biofilm_model.npz")
    p.add_argument("--out", default="results")
    p.add_argument("--window", type=int)
    p.add_argument("--overlap-thr", type=float)
    p.add_argument("--scale", type=float)

    v = sub.add_parser("validate", help="held-out + synthetic co-culture validation")
    v.add_argument("--bsub", nargs="+", required=True)
    v.add_argument("--ecoli", nargs="+", required=True)
    v.add_argument("--out", default="validation")
    v.add_argument("--scale", type=float, default=1.0)

    a = ap.parse_args(argv)
    if a.cmd == "train":
        fo, meta = train(a.bsub, a.ecoli, a.scale)
        meta.window, meta.overlap_thr = a.window, a.overlap_thr
        os.makedirs(os.path.dirname(a.model) or ".", exist_ok=True)
        save_model(a.model, fo, meta)
        print(f"saved {a.model}  ({meta.n_bsub_px} B. sub px, {meta.n_ecoli_px} E. coli px); "
              f"held-out hit rate B. sub {meta.tpr:.3f}, E. coli called B. sub {meta.fpr:.3f}")
    elif a.cmd == "predict":
        predict_files(a.model, a.images, a.out, a.window, a.overlap_thr, a.scale)
    elif a.cmd == "validate":
        from validate import run_validation
        run_validation(a.bsub, a.ecoli, a.out, a.scale)


if __name__ == "__main__":
    main()
