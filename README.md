# Museum Art Guide — neural-network components

Final project, *Neural Networks & Deep Learning*.

The (fictional) Museum wants a guide, on visitors' phones and on gallery kiosks, that recognises the
**style** of the painting the visitor points the camera at and tells the story of that movement.
This repository builds, **diagnoses** and compares the classifiers behind that feature, and answers
the director's question:

> *The kiosks run on modest CPUs, the team is small, the budget is limited. Is a neural network worth
> it in production, or would a simpler classical model do the job?*

Every number in the report is produced by a script of this repository, with confidence intervals.

| Deliverable | Where |
|---|---|
| Feed-forward network in PyTorch, built from primitives | `src/museum_guide/mlp.py`, `scripts/02_train_mlp.py` |
| Training monitoring and failure diagnosis | `src/museum_guide/training.py`, `scripts/03_diagnose_mlp.py` |
| Pretrained CNN (ResNet-18) | `src/museum_guide/resnet.py`, `scripts/04_train_resnet.py` |
| Classical baseline + CPU cost → recommendation | `scripts/01_baseline.py`, `scripts/05_benchmark_cpu.py`, `scripts/06_summary.py` |

## Results

5 styles × 300 images, 128 px; artist-grouped split 1 001 / 263 / 236 (train / val / test). Chance = 20 %.
Training on Colab (T4 GPU); CPU benchmark on Colab's 2-vCPU machine, one photo, 1 thread.

| model | test accuracy [95 % CI] | macro-F1 | CPU latency | size | training |
|---|---|---|---|---|---|
| Logistic regression — raw pixels 32×32 | 35.2 % [28.8–41.5] | 0.34 | 1.9 ms | 0.2 MB | 40 s CPU |
| Logistic regression — hand-crafted features | 62.7 % [56.8–69.1] | 0.62 | 16.7 ms | 0.1 MB | 4 s CPU |
| **MLP from scratch** — raw pixels 32×32 | 35.7 % ± 1.1 (3 seeds) | 0.32 | 1.3 ms | 6.9 MB | 1 s |
| MLP from scratch — hand-crafted features | 57.6 % ± 1.5 (3 seeds) | 0.58 | ~17 ms | — | 1 s |
| ResNet-18 frozen + linear head | 69.9 % [64.0–75.4] | 0.70 | 23.1 ms | 44.8 MB | 1 s (cached embeddings) |
| **ResNet-18 frozen + logistic regression** | **76.3 % [70.8–81.4]** | 0.76 | 23.1 ms | 44.8 MB | seconds, CPU |
| ResNet-18 fine-tuned (`layer4` + head) | 77.1 % [71.6–82.6] | 0.77 | 23.1 ms | 44.8 MB | 63 s GPU |

![quality vs cost](results/tradeoff.png)

**What the numbers say**

1. **The from-scratch MLP brings nothing over a linear model on the same input**: 35.7 % vs 35.2 % on pixels,
   57.6 % vs 62.7 % on hand-crafted features. On raw pixels it overfits after 2–3 epochs (train accuracy keeps
   rising to 83 %, validation stays at ~40 %, validation loss climbs: `results/02_mlp/pixels32/training_history.png`).
   A fully connected network has no notion of neighbouring pixels and 1.7 M weights for 1 001 images.
2. **Knowledge transferred from ImageNet is what makes the difference**: +14 points over the best classical model
   (the confidence intervals do not overlap), top-2 accuracy 91 %.
3. **Fine-tuning is not worth it here**: 77.1 % vs 76.3 % for the frozen backbone + logistic regression, well within
   the confidence intervals, for a GPU and 60× more training time.
4. **Cost on CPU is acceptable**: 23 ms per photo on one thread at 128 px (55 ms at 224 px), far below the ~100 ms
   perceived as instantaneous; the hand-crafted pipeline is not much cheaper (17 ms, feature extraction dominates).
5. **Remaining errors** are mostly Pop Art → Cubism (flat colour areas, geometric shapes) and Impressionism ↔ Baroque.
6. **Diagnosis** (`results/03_diagnosis/summary.md`): every injected failure shows its expected, measured symptom —
   zero init: zero hidden gradients and 100 % dead units; deep sigmoid: first-layer gradient 7 000× smaller than the
   output layer's; LR 1e-6: update/weight ratio 1e-5; LR 0.1: 50 % dead ReLUs, chance-level accuracy; shuffled labels:
   100 % train vs 22 % test; overfit-one-batch: 100 %. The LR range test puts the optimum at 1e-3 (the value used).
   The same "initial loss = ln K" check flags the ResNet linear head (initial loss 2.8 instead of 1.61, unscaled
   output initialisation), which explains part of its gap to the logistic regression on the same embeddings.

**Recommendation to the director.** Yes, a neural network is worth it — **a pretrained one, not one trained from
scratch**. Deploy ResNet-18 as a frozen feature extractor followed by a logistic regression: 76 % vs 63 % for the best
classical model, 23 ms per photo on one CPU core, 45 MB, and retraining (e.g. when styles are added) takes seconds on
a laptop, with no GPU to buy. Fine-tuning does not pay for itself on this data.

**Limits (stated, not hidden).** Only 15 named painters in the subset: Impressionism and Pop Art are 100 % "Unknown
Artist" and Baroque 96 %, so the artist-grouped split cannot guarantee that test paintings come from unseen painters;
consistently, the random-vs-artist split comparison is small and inconsistent (+4.8, +3.0 and −4.0 points, all inside
the confidence intervals) and test accuracy may be optimistic. Other limits: 5 contrasted styles, 236 test images
(±6 points), WikiArt scans rather than visitors' photos (reflections, angles), latency measured on a cloud CPU rather
than the kiosk hardware.

## Quick start

```bash
git clone https://github.com/<owner>/museum-art-guide.git && cd museum-art-guide
pip install -r requirements.txt && pip install -e .
python -m pytest -q                      # 18 tests, ~10 s, no download

bash scripts/build_dataset.sh            # WikiArt subset -> data/wikiart_style5 (~4 GB streamed, 8 MB kept)
bash scripts/run_all.sh                  # steps 0-6 -> results/
```

On Google Colab: open `notebooks/colab_run_all.ipynb` (GPU runtime) — it does all of the above.

Smoke test of the whole pipeline in ~4 minutes, without downloading anything (synthetic data, numbers
meaningless): `python scripts/make_fake_dataset.py --out data/fake --per-class 80 && DATA=data/fake SIZE=64 RESULTS=/tmp/r QUICK=1 bash scripts/run_all.sh`

## Pipeline

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| data | `build_dataset.sh` | 5 contrasted styles (Impressionism, Baroque, Cubism, Ukiyo-e, Pop Art) × 300 images, 128 px, ≤ 50 images per named artist | `data/wikiart_style5/` (git-ignored) |
| 0 | `00_explore.py` | class / artist statistics, artist-grouped split | `results/00_explore/eda.json` |
| 1 | `01_baseline.py` | logistic regression on raw pixels and on hand-crafted features (HSV histogram, colour moments, HOG, LBP) | `results/01_baseline/` |
| 2 | `02_train_mlp.py` | from-scratch MLP: sweep on validation, then 3 seeds | `results/02_mlp/<features>/` |
| 3 | `03_diagnose_mlp.py` | 11 controlled failure scenarios + learning-rate range test | `results/03_diagnosis/summary.md` |
| 4 | `04_train_resnet.py` | ResNet-18: frozen backbone (+ linear head, + logistic regression) and fine-tuning of `layer4` | `results/04_resnet/` |
| 5 | `05_benchmark_cpu.py` | end-to-end CPU latency for one photo (1/2/4 threads), size, parameters | `results/05_benchmark/` |
| 6 | `06_summary.py` | quality vs cost table and figure | `results/summary.md`, `results/tradeoff.png` |

## Methodological choices (and why)

* **Split grouped by artist.** WikiArt is ordered by painter and a painter has a recognisable hand. With
  a random split the same painter is in train and test and the model can recognise *Monet* instead of
  *Impressionism*. The guide will meet unseen artists, so no named artist is shared between
  train/val/test (`data.grouped_stratified_split`, stratified by class × named/unknown artist).
  `--split random` exists only to **measure** the leakage (diagnosis scenario `random_split`).
* **Validation for every choice, test once.** Hyper-parameters, regularisation strength and early
  stopping use the validation split; the test split is only used for the final numbers.
* **Uncertainty.** ~225 test images → about ±6 accuracy points. Every test metric carries a 95 %
  bootstrap interval; MLP results are averaged over 3 seeds (mean ± std).
* **Network from primitives.** `Linear` is `x @ W.T + b` on `nn.Parameter` with explicit He/Xavier
  initialisation; dropout is written by hand; the training loop (optimiser step, early stopping,
  monitoring) is ours. Only autograd, the optimiser and the cross-entropy come from PyTorch.
* **Monitoring that diagnoses.** Initial loss vs ln(K), gradient norm and update/weight ratio per layer,
  dead/saturated units; each failure scenario of step 3 shows the symptom it produces.
* **Colour-preserving augmentation.** Only crops and flips for the CNN: colour is a style cue.
* **Cost measured end to end.** Feature extraction is charged to the classical models; latency is the
  median over 100 runs on CPU after warm-up.

## Repository layout

```
src/museum_guide/   data.py  features.py  baseline.py  mlp.py  training.py  resnet.py
                    evaluation.py  plots.py  experiments.py  common.py
scripts/            make_wikiart_subset.py (provided)  build_dataset.sh  make_fake_dataset.py
                    00_explore.py ... 06_summary.py  run_all.sh
notebooks/          colab_run_all.ipynb
tests/              unit tests (pytest)
results/            JSON + figures produced by the scripts (committed: they are the evidence)
docs/               presentation plan, Q&A preparation, team workflow (French)
```

## Data and licence

WikiArt images belong to their rights holders (dataset card: "license: unknown"). They are **never**
committed: `data/` and the figures showing paintings are git-ignored. The exact subset is reproducible
from `scripts/build_dataset.sh` (pinned dataset revision, seed 0).

## Team

Two members — see `docs/CONTRIBUTING.md` for the split of work; the commit history shows who did what.
