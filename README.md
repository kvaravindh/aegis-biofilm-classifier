# AEGIS Biofilm Species Classifier

Estimates how much of a crystal-violet-stained, co-inoculated biofilm is *B. subtilis* NCIB 3610,
how much is *E. coli* MG1655, and how much is overlap (both intermixed), from brightfield images.

**Use it:** https://kvaravindh.github.io/aegis-biofilm-classifier/

Images dropped into the site are analysed in the browser and never uploaded.

NUS-SPS iGEM 2026 · AEGIS

---

## Adding training photos (team)

The model learns from monoculture photos in two folders:

| Folder | Put here |
|---|---|
| `training/bsub/` | pure *B. subtilis* NCIB 3610 wells |
| `training/ecoli/` | pure *E. coli* MG1655 wells |

1. On the site, click **Add B. subtilis photos** or **Add E. coli photos**. You can also open the
   folder on GitHub and choose **Add file → Upload files**.
2. Drag in PNG, JPEG or TIFF photos (many at once is fine, up to 100 per upload) and click **Commit changes**.
3. The **Retrain model and publish site** action runs automatically (Actions tab). After a few
   minutes, the site uses the new model.

Anyone added as a collaborator can commit photos directly (repo **Settings → Collaborators**).
Anyone else gets a pull request from the upload page, which a collaborator merges.

**Photo rules**

- Pure monoculture wells only. Every pixel of a photo is taken as that species.
- Same crystal-violet protocol, objective and camera zoom as the co-culture images.
- About 750–950 px across a field. Photos over 3.2 megapixels are skipped; resize them first.
- 3–5 fields per well, from at least 3 wells and 2 days per species. Include dense and sparse areas.

### Checking a retrain

Each run's summary page (Actions → the run) shows:

- how many photos were used;
- the share of pixels called correctly on photos each model never saw;
- any photo scoring below 70%, which is usually a wrong label, blur or a different zoom;
- any skipped photo.

The same figures are saved in `model/metrics.json`. To remove a photo, click **Remove** under it on
the site and commit on GitHub; the model retrains without it.

## How it works

1. **Stain map.** Optical density of the green channel: `OD = −log10(G / G_bg)`. Hue is ignored.
2. **Biofilm mask.** Otsu threshold on OD, limited to 0.08–0.25. Unstained background is excluded
   from every percentage.
3. **Morphology features.** 33 per pixel at σ = 1, 2, 4, 8 px, after local contrast normalisation:
   - ridges and elongation;
   - edges and bundle alignment;
   - texture.
4. **Random forest.** 60 trees, depth 14, trained on monoculture pixels. Held-out error rates
   correct the counts: `f′ = (f − fpr) / (tpr − fpr)`.
5. **Local mix.** In a 15 px radius around each biofilm pixel, the local *B. subtilis* share f′
   decides the call:
   - **E. coli** if f′ ≤ 0.25;
   - **B. subtilis** if f′ ≥ 0.75;
   - **overlap** in between.

The site (`site/engine.js`) and the Python code (`bfcore.py`) implement the same steps. On the
reference images they give identical calls, pixel for pixel.

## Python

```bash
pip install -r requirements.txt
python biofilm_cv.py predict --model model/biofilm_model.npz --out results/ well_A1.png well_A2.png
python scripts/train.py          # retrain from training/ (what the Action runs)
python site/build_site.py        # build the site into _site/
```

## Repository layout

| Path | What it is |
|---|---|
| `training/bsub`, `training/ecoli` | Training photos |
| `model/biofilm_model.npz`, `model/metrics.json` | Current model and its held-out scores, written by the Action |
| `scripts/train.py` | Trains the one team model from `training/` |
| `site/` | Web page template, browser engine, build script |
| `bfcore.py`, `biofilm_cv.py`, `validate.py` | Python core, command-line tool, synthetic validation |
| `.github/workflows/train-and-publish.yml` | Retrains on every push to `main` and deploys to GitHub Pages |

## Limitations

- **Imaging must match.** A different zoom or resolution makes cells look a different shape to the
  model.
- **Single-cell mixing isn't resolved.** Mixing at the scale of single cells is below what
  brightfield can separate at this magnification, so it is reported as overlap.
- **Validate with a known answer.** Before reporting co-culture numbers, check the tool on one
  co-culture with fluorescent reporters (e.g. GFP and mCherry) or a Gram stain.
