"""
Validation for biofilm_cv.

A. Spatial hold-out: train on one half of each reference image, test on the other half
   (both orientations, both directions = 4 folds). A 32 px gap is left between halves.
B. Synthetic co-cultures with known truth: held-out B. subtilis and E. coli regions are
   spliced together through a known mask — either large domains (segregated growth) or a
   fine patchwork (intermixed growth) — at several mixing ratios. The model's estimates
   of composition and overlap are compared with the truth computed from the mask.
"""
from __future__ import annotations

import json
import os

import warnings

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

import biofilm_cv as bc

GAP = 32
warnings.filterwarnings("ignore", category=FutureWarning)


def halves(shape, axis, which):
    H, W = shape
    m = np.zeros(shape, bool)
    n = (H if axis == 0 else W)
    cut = n // 2
    sl = slice(0, cut - GAP // 2) if which == 0 else slice(cut + GAP // 2, n)
    if axis == 0:
        m[sl, :] = True
    else:
        m[:, sl] = True
    return m


def crop_half(a, axis, which):
    n = a.shape[axis]
    cut = n // 2
    sl = slice(0, cut - GAP // 2) if which == 0 else slice(cut + GAP // 2, n)
    return a[sl] if axis == 0 else a[:, sl]


def truth_stats(true_b, mask, window, thr):
    k = morphology.disk(window).astype(np.float32)
    nb = ndi.convolve((true_b & mask).astype(np.float32), k, mode="reflect")
    nm = ndi.convolve(mask.astype(np.float32), k, mode="reflect")
    fb = np.where(nm > 0, nb / np.maximum(nm, 1e-6), 0)
    n = mask.sum()
    return dict(bsub_total_pct=100 * fb[mask].sum() / n,
                bsub_only_pct=100 * (mask & (fb >= 1 - thr)).sum() / n,
                ecoli_only_pct=100 * (mask & (fb <= thr)).sum() / n,
                overlap_pct=100 * (mask & (fb > thr) & (fb < 1 - thr)).sum() / n)


def blob_mask(shape, frac, rng, sigma):
    z = ndi.gaussian_filter(rng.standard_normal(shape), sigma)
    return z > np.quantile(z, 1 - frac)


def patch_mask(shape, frac, rng, size):
    gh, gw = -(-shape[0] // size), -(-shape[1] // size)
    g = rng.random((gh, gw)) < frac
    return np.kron(g, np.ones((size, size), bool))[: shape[0], : shape[1]]


def run_validation(bsub_paths, ecoli_paths, out_dir, scale=1.0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(1)
    meta = bc.ModelMeta()
    W_, T_ = meta.window, meta.overlap_thr

    B = [bc.prep(p, scale) for p in bsub_paths]
    E = [bc.prep(p, scale) for p in ecoli_paths]

    report = {"holdout": [], "synthetic": []}
    synth_examples = []
    for axis in (0, 1):
        for train_half in (0, 1):
            test_half = 1 - train_half
            rf, tpr, fpr, _, _ = bc.fit_calibrated(B, E, lambda sh: halves(sh, axis, train_half), 30000)

            # A. held-out halves, classified as standalone images (features recomputed on the crop)
            fold = {"axis": "rows" if axis == 0 else "cols", "train_half": train_half}
            for name, data, truth in (("bsub", B, bc.BSUB), ("ecoli", E, bc.ECOLI)):
                accs, comp, ovl = [], [], []
                for od, _, _ in data:
                    odc = crop_half(od, axis, test_half)
                    mc, _ = bc.biomass_mask(odc)
                    r = bc.classify(rf, odc, mc, W_, T_, tpr, fpr)
                    hard = (r["prob"] >= 0.5)[mc]
                    accs.append(float(np.mean(hard == (truth == bc.BSUB))))
                    comp.append(r["stats"]["bsub_total_pct"] if truth == bc.BSUB else r["stats"]["ecoli_total_pct"])
                    ovl.append(r["stats"]["overlap_pct"])
                fold[f"{name}_pixel_acc"] = float(np.mean(accs))
                fold[f"{name}_correct_composition_pct"] = float(np.mean(comp))
                fold[f"{name}_false_overlap_pct"] = float(np.mean(ovl))
            report["holdout"].append(fold)

            # B. synthetic co-cultures from the held-out halves
            odb = crop_half(B[0][0], axis, test_half)
            ode = crop_half(E[0][0], axis, test_half)
            h, w = min(odb.shape[0], ode.shape[0]), min(odb.shape[1], ode.shape[1])
            odb, ode = odb[:h, :w], ode[:h, :w]
            for kind in ("domains", "intermixed"):
                for frac in (0.2, 0.5, 0.8):
                    # domains: smooth blobs ~100+ px across; intermixed: random 25 px patches (microcolony scale)
                    M = blob_mask((h, w), frac, rng, 25) if kind == "domains" else patch_mask((h, w), frac, rng, 25)
                    od = np.where(M, odb, ode)
                    mask, _ = bc.biomass_mask(od)
                    r = bc.classify(rf, od, mask, W_, T_, tpr, fpr)
                    t = truth_stats(M, mask, W_, T_)
                    row = {"fold": f"{fold['axis']}-{train_half}", "kind": kind, "target_bsub_frac": frac,
                           "tpr": round(tpr, 3), "fpr": round(fpr, 3)}
                    for k in ("bsub_total_pct", "bsub_only_pct", "ecoli_only_pct", "overlap_pct"):
                        row[f"true_{k}"] = round(float(t[k]), 1)
                        row[f"pred_{k}"] = round(float(r["stats"][k]), 1)
                    report["synthetic"].append(row)
                    if axis == 0 and train_half == 0 and frac == 0.5:
                        synth_examples.append((kind, od, M, mask, r))

    # summary
    hold = report["holdout"]
    syn = report["synthetic"]
    summ = {
        "holdout_pixel_acc_bsub": np.mean([f["bsub_pixel_acc"] for f in hold]),
        "holdout_pixel_acc_ecoli": np.mean([f["ecoli_pixel_acc"] for f in hold]),
        "holdout_composition_correct_bsub": np.mean([f["bsub_correct_composition_pct"] for f in hold]),
        "holdout_composition_correct_ecoli": np.mean([f["ecoli_correct_composition_pct"] for f in hold]),
        "holdout_false_overlap_bsub": np.mean([f["bsub_false_overlap_pct"] for f in hold]),
        "holdout_false_overlap_ecoli": np.mean([f["ecoli_false_overlap_pct"] for f in hold]),
    }
    for kind in ("domains", "intermixed"):
        rs = [r for r in syn if r["kind"] == kind]
        summ[f"synthetic_{kind}_composition_MAE_pts"] = np.mean([abs(r["true_bsub_total_pct"] - r["pred_bsub_total_pct"]) for r in rs])
        summ[f"synthetic_{kind}_overlap_MAE_pts"] = np.mean([abs(r["true_overlap_pct"] - r["pred_overlap_pct"]) for r in rs])
    report["summary"] = {k: round(float(v), 2) for k, v in summ.items()}
    with open(os.path.join(out_dir, "validation.json"), "w") as f:
        json.dump(report, f, indent=1)

    # figure: true vs predicted
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for kind, mk in (("domains", "o"), ("intermixed", "s")):
        rs = [r for r in syn if r["kind"] == kind]
        ax[0].scatter([r["true_bsub_total_pct"] for r in rs], [r["pred_bsub_total_pct"] for r in rs], marker=mk, label=kind, alpha=.8)
        ax[1].scatter([r["true_overlap_pct"] for r in rs], [r["pred_overlap_pct"] for r in rs], marker=mk, label=kind, alpha=.8)
    for a, t in zip(ax, ("B. subtilis share of biomass (%)", "Overlap area (% of biomass)")):
        a.plot([0, 100], [0, 100], "k--", lw=1); a.set_xlim(0, 100); a.set_ylim(0, 100)
        a.set_xlabel(f"True {t}"); a.set_ylabel(f"Predicted {t}"); a.legend(); a.set_aspect("equal")
    fig.suptitle("Synthetic co-cultures built from held-out image halves (4 folds × 2 layouts × 3 ratios)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "synthetic_true_vs_pred.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(len(synth_examples), 3, figsize=(14, 4.2 * len(synth_examples)))
    for i, (kind, od, M, mask, r) in enumerate(synth_examples):
        ax[i, 0].imshow(od, cmap="gray_r"); ax[i, 0].set_title(f"Synthetic ({kind}), stain OD")
        tl = np.zeros(M.shape, np.uint8); tl[mask & M] = 1; tl[mask & ~M] = 2
        ax[i, 1].imshow(bc.overlay(np.dstack([255 - od.clip(0, 1) * 255] * 3).astype(np.float32), tl)); ax[i, 1].set_title("Truth (blue B. sub, orange E. coli)")
        ax[i, 2].imshow(bc.overlay(np.dstack([255 - od.clip(0, 1) * 255] * 3).astype(np.float32), r["labels"]))
        s = r["stats"]; ax[i, 2].set_title(f"Predicted: Bs {s['bsub_only_pct']:.0f}% | Ec {s['ecoli_only_pct']:.0f}% | overlap {s['overlap_pct']:.0f}%")
        for a in ax[i]: a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "synthetic_examples.png"), dpi=100); plt.close(fig)

    print(json.dumps(report["summary"], indent=1))
    return report
