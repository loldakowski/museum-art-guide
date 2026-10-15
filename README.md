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

*To be filled from `results/summary.md` after the full run (`bash scripts/run_all.sh`).*

![quality vs cost](results/tradeoff.png)

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
