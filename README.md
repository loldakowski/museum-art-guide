# Museum Art Guide

Final project for the Neural Networks & Deep Learning course.

A (fictional) museum wants a guide that recognises the style of a painting from a visitor's photo. We built
and compared several classifiers for this task, and tried to answer the director's question: the kiosks only
have modest CPUs, so is a neural network worth it, or would a classical model be enough?

The three deliverables:

- a feed-forward network written from PyTorch primitives: `src/museum_guide/mlp.py`, `scripts/02_train_mlp.py`
- training monitoring and failure diagnosis: `src/museum_guide/training.py`, `scripts/03_diagnose_mlp.py`
- a pretrained ResNet-18: `src/museum_guide/resnet.py`, `scripts/04_train_resnet.py`

We also trained a logistic regression baseline (`scripts/01_baseline.py`) and measured CPU latency
(`scripts/05_benchmark_cpu.py`) to compare everything on the same test set.

## Data

5 styles from WikiArt (Impressionism, Baroque, Cubism, Ukiyo-e, Pop Art), 300 images each, resized to 128 px.
The subset is built with the script provided for the course (`scripts/make_wikiart_subset.py`), see
`scripts/build_dataset.sh` for the exact command. Images are not included in the repo (unknown licence).

Train / val / test: 1001 / 263 / 236 images. The split is grouped by artist, so a named painter never appears
in two splits (otherwise the model could learn to recognise the painter instead of the style).

## Results

Accuracy on the test set (chance = 20 %), with 95 % bootstrap intervals. Latency is the median time to
classify one photo on CPU with one thread (Colab machine, 2 vCPU).

| Model | Test accuracy | Latency | Size |
|---|---|---|---|
| Logistic regression, raw pixels (32x32) | 35.2 % [28.8, 41.5] | 1.9 ms | 0.2 MB |
| Logistic regression, hand-crafted features (colour, HOG, LBP) | 62.7 % [56.8, 69.1] | 16.7 ms | 0.1 MB |
| MLP, raw pixels (mean of 3 seeds) | 35.7 % ± 1.1 | 1.3 ms | 6.9 MB |
| MLP, hand-crafted features (mean of 3 seeds) | 57.6 % ± 1.5 | ~17 ms | - |
| ResNet-18 frozen + linear head | 69.9 % [64.0, 75.4] | 23.1 ms | 44.8 MB |
| ResNet-18 frozen + logistic regression | 76.3 % [70.8, 81.4] | 23.1 ms | 44.8 MB |
| ResNet-18 fine-tuned (layer4 + head) | 77.1 % [71.6, 82.6] | 23.1 ms | 44.8 MB |

![quality vs latency](results/tradeoff.png)

Main observations:

- The MLP does not do better than a logistic regression on the same inputs. On raw pixels it overfits after
  2-3 epochs: training accuracy keeps going up (83 %) while validation stays around 40 %
  (`results/02_mlp/pixels32/training_history.png`).
- The pretrained ResNet is clearly better than the best classical model (+14 points, the intervals don't overlap).
  Top-2 accuracy is 91 %.
- Fine-tuning gives almost nothing compared to the frozen backbone (77.1 % vs 76.3 %, inside the intervals) and
  needs a GPU.
- 23 ms per photo on one CPU core at 128 px (55 ms at 224 px), which is fine for a kiosk.
- Most errors: Pop Art predicted as Cubism, and Impressionism / Baroque confusions.

Failure diagnosis (`results/03_diagnosis/summary.md`): we broke the MLP on purpose to check that the monitoring
shows each problem. Zero initialisation gives zero gradients in the hidden layers and 100 % dead units, a deep
sigmoid network has first-layer gradients about 7000x smaller than the output layer, lr = 1e-6 gives an
update/weight ratio around 1e-5, lr = 0.1 kills half of the ReLUs, and with shuffled labels the model reaches
100 % on train but stays at chance on test. The LR range test gives 1e-3 as the best value, which is what we use.
The same initial loss check (should be ln 5 = 1.61) shows that the ResNet linear head starts at 2.8, which
probably explains why it does worse than the logistic regression on the same features.

## Recommendation

A neural network is worth it, but a pretrained one, not one trained from scratch. We would deploy ResNet-18 as
a frozen feature extractor followed by a logistic regression: 76 % accuracy against 63 % for the best classical
model, 23 ms per photo on CPU, 45 MB, and it can be retrained in a few seconds on a laptop (no GPU needed).

Limits: only 15 named painters in the subset (Impressionism and Pop Art are entirely "Unknown Artist"), so we
cannot really guarantee that the test set only has unseen painters, and comparing the random and artist splits
gave small, inconsistent differences. Test accuracy may therefore be a bit optimistic. The test set is small
(about ±6 points), WikiArt scans are not visitor photos, and latency was measured on a cloud CPU.

## How to run

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest -q             # unit tests, no download needed

bash scripts/build_dataset.sh   # downloads the subset into data/wikiart_style5
bash scripts/run_all.sh         # runs steps 0 to 6, outputs in results/
```

Or open `notebooks/colab_run_all.ipynb` in Google Colab (GPU runtime) and run all cells.

## Structure

```
src/museum_guide/   data loading and split, features, baseline, MLP, training loop, ResNet, metrics, plots
scripts/            00_explore.py ... 06_summary.py, run_all.sh, dataset scripts
notebooks/          Colab notebook
tests/              unit tests
results/            metrics (JSON) and figures from the full run
```
