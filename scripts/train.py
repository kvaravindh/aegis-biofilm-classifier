#!/usr/bin/env python3
"""
Train the one team model from the photos in training/bsub and training/ecoli.

- Each photo is checked by a model that never saw it (photos split into up to 4 groups,
  each group held out in turn). Those held-out scores calibrate the composition estimates
  and are written to model/metrics.json and to the GitHub Actions job summary.
- With only one photo of a species, it falls back to holding out halves of each photo.
- The final model is trained on every photo and saved to model/biofilm_model.npz.

Run locally:  python scripts/train.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import biofilm_cv as bc  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
MAX_PX = 3.2e6
PER_CLASS = 40000
SPECIES = {"bsub": bc.BSUB, "ecoli": bc.ECOLI}
NAMES = {"bsub": "B. subtilis NCIB 3610", "ecoli": "E. coli MG1655"}


def photos(sp):
    d = ROOT / "training" / sp
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS) if d.exists() else []


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    items, skipped = [], []
    lists = {sp: photos(sp) for sp in SPECIES}
    for sp, ps in lists.items():
        per = int(np.clip(PER_CLASS // max(1, len(ps)), 2000, 12000))
        for p in ps:
            try:
                rgb = bc.load_rgb(str(p))
                if rgb.shape[0] * rgb.shape[1] > MAX_PX:
                    skipped.append((p, f"{rgb.shape[1]}×{rgb.shape[0]} px is too large; resize to the reference zoom (~750–950 px across)"))
                    continue
                od = bc.stain_od(rgb)
                mask, _ = bc.biomass_mask(od)
                F = bc.image_features(od)
                if mask.sum() < 200:
                    skipped.append((p, "almost no stained biofilm found"))
                    continue
                items.append(dict(path=p, sp=sp, lab=SPECIES[sp], od=od, mask=mask, F=F, X=bc._sample(F, mask, per, rng)))
                print(f"features: {p.relative_to(ROOT)}  ({len(items[-1]['X'])} pixels)", flush=True)
            except Exception as e:  # unreadable file
                skipped.append((p, f"could not be read ({e.__class__.__name__})"))
    B = [i for i in items if i["sp"] == "bsub"]
    E = [i for i in items if i["sp"] == "ecoli"]
    if not B or not E:
        sys.exit("Need at least one usable photo in each of training/bsub and training/ecoli.")

    k = min(4, len(B), len(E))
    per_photo = {}
    if k >= 2:
        method = f"photo-level hold-out ({k} groups)"
        for arr in (B, E):
            for j, it in enumerate(arr):
                it["fold"] = j % k
        for f in range(k):
            fo = bc.fit_rf(np.vstack([i["X"] for i in B if i["fold"] != f]), np.vstack([i["X"] for i in E if i["fold"] != f]))
            for it in items:
                if it["fold"] == f:
                    called_b = fo.predict(it["X"]) >= 0.5
                    per_photo[str(it["path"].relative_to(ROOT))] = float(np.mean(called_b == (it["lab"] == bc.BSUB)))
            print(f"held-out group {f + 1}/{k} done", flush=True)
        accB = float(np.mean([per_photo[str(i["path"].relative_to(ROOT))] for i in B]))
        accE = float(np.mean([per_photo[str(i["path"].relative_to(ROOT))] for i in E]))
        tpr, fpr = accB, 1 - accE
        final = bc.fit_rf(np.vstack([i["X"] for i in B]), np.vstack([i["X"] for i in E]))
    else:
        method = "half-photo hold-out (add a second photo of each species for photo-level checks)"
        final, tpr, fpr, _, _ = bc.fit_calibrated([(i["od"], i["mask"], i["F"]) for i in B], [(i["od"], i["mask"], i["F"]) for i in E])
        accB, accE = tpr, 1 - fpr

    meta = bc.ModelMeta(tpr=tpr, fpr=fpr, n_bsub_px=int(sum(len(i["X"]) for i in B)), n_ecoli_px=int(sum(len(i["X"]) for i in E)),
                        bsub_refs=[i["path"].name for i in B], ecoli_refs=[i["path"].name for i in E])
    (ROOT / "model").mkdir(exist_ok=True)
    bc.save_model(str(ROOT / "model" / "biofilm_model.npz"), final, meta)
    metrics = dict(method=method, held_out_accuracy_bsub=round(accB, 4), held_out_accuracy_ecoli=round(accE, 4),
                   tpr=round(tpr, 4), fpr=round(fpr, 4), photos={"bsub": len(B), "ecoli": len(E)},
                   per_photo_held_out={k_: round(v, 4) for k_, v in sorted(per_photo.items())},
                   skipped={str(p.relative_to(ROOT)): why for p, why in skipped}, tree_nodes=int(len(final.feat)))
    (ROOT / "model" / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")

    # report
    lines = ["## Biofilm model retrained", "",
             f"Photos used: **{len(B)}** {NAMES['bsub']}, **{len(E)}** {NAMES['ecoli']} · checked by {method}", "",
             "| Species | Held-out pixels called correctly |", "|---|---|",
             f"| {NAMES['bsub']} | {accB:.0%} |", f"| {NAMES['ecoli']} | {accE:.0%} |", ""]
    weak = [(p, a) for p, a in per_photo.items() if a < 0.7]
    if per_photo:
        lines += ["<details><summary>Per-photo held-out accuracy</summary>", "", "| Photo | Held-out |", "|---|---|"]
        lines += [f"| `{p}` | {a:.0%}{' ⚠️' if a < 0.7 else ''} |" for p, a in sorted(per_photo.items(), key=lambda x: x[1])]
        lines += ["", "</details>", ""]
    if weak:
        lines += [f"**{len(weak)} photo(s) score below 70%**: check labels, focus and zoom: " + ", ".join(f"`{p}`" for p, _ in weak), ""]
    if skipped:
        lines += ["**Skipped photos:**", ""] + [f"- `{p.relative_to(ROOT)}`: {why}" for p, why in skipped] + [""]
    lines.append(f"Trained in {time.time() - t0:.0f} s.")
    report = "\n".join(lines) + "\n"
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(report)


if __name__ == "__main__":
    main()
