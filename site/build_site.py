#!/usr/bin/env python3
"""Build _site/: index.html (model, engine and examples inlined) plus training-photo thumbnails."""
from __future__ import annotations

import base64
import gzip
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SITE = Path(__file__).resolve().parent
OUT = ROOT / "_site"
sys.path.insert(0, str(ROOT))
import biofilm_cv as bc  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def pack_model():
    fo, meta = bc.load_model(str(ROOT / "model" / "biofilm_model.npz"))
    parts = [("feat", fo.feat.astype(np.uint8)), ("thr", fo.thr.astype("<f8")), ("left", fo.left.astype("<u4")),
             ("right", fo.right.astype("<u4")), ("roots", fo.roots.astype("<u4"))]
    buf, header = bytearray(), {}
    for name, arr in parts:
        while len(buf) % 8:
            buf.append(0)
        header[name] = [len(buf), int(arr.size)]
        buf += arr.tobytes()
    m = asdict(meta)
    return base64.b64encode(gzip.compress(bytes(buf), 9)).decode(), {
        "header": header, "meta": {"tpr": m["tpr"], "fpr": m["fpr"], "window": m["window"], "overlap_thr": m["overlap_thr"]}}


def thumbs():
    out = {}
    for sp in ("bsub", "ecoli"):
        out[sp] = []
        d = ROOT / "training" / sp
        if not d.exists():
            continue
        (OUT / "thumbs" / sp).mkdir(parents=True, exist_ok=True)
        for p in sorted(x for x in d.iterdir() if x.suffix.lower() in EXTS):
            try:
                im = Image.open(p).convert("RGB")
                im.thumbnail((280, 280))
                rel = f"thumbs/{sp}/{p.stem}.jpg"
                im.save(OUT / rel, quality=82)
                out[sp].append({"name": p.name, "thumb": rel})
            except Exception:
                pass
    return out


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    b64 = lambda p: base64.b64encode((SITE / p).read_bytes()).decode()
    model_b64, model_hdr = pack_model()
    training = {"repo": os.environ.get("GITHUB_REPOSITORY", "kvaravindh/aegis-biofilm-classifier"),
                "branch": os.environ.get("GITHUB_REF_NAME", "main"), "photos": thumbs()}
    engine = (SITE / "engine.js").read_text()
    assert "</script" not in engine
    t = (SITE / "template.html").read_text()
    rep = {"{{CSS}}": (SITE / "style.css").read_text(), "{{ENGINE}}": engine, "{{MODEL_B64}}": model_b64,
           "{{MODEL_HEADER}}": json.dumps(model_hdr), "{{TRAINING_JSON}}": json.dumps(training).replace("</", "<\\/")}
    for k, v in rep.items():
        assert k in t, k
        t = t.replace(k, v)
    (OUT / "index.html").write_text(t)
    (OUT / ".nojekyll").write_text("")
    print(f"built _site/index.html ({len(t) / 1e6:.1f} MB), {sum(len(v) for v in training['photos'].values())} training thumbnails")


if __name__ == "__main__":
    main()
