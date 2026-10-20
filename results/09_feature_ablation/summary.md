# Hand-crafted feature ablation (validation set only)

Train 1032, validation 223 images; the test set is not used. C chosen on validation, so every accuracy is optimistic by the same selection bias.
Extraction time: one 128 px image at a time, median over 100 train images x 3 repeats (arm64, 10 CPUs).

| variant | features | best C | val acc [95% CI] | diff vs default [paired 95% CI] | McNemar p (Holm) | extraction ms/image (median, p95) |
|---|---|---|---|---|---|---|
| all blocks (default) | 1975 | 0.001 | 0.583 [0.520, 0.646] | - | - | 2.91, 3.02 |
| hsv alone | 192 | 0.01 | 0.498 [0.426, 0.561] | -0.085 [-0.170, +0.000] | 0.0671 (0.603) | 1.34, 1.43 |
| moments alone | 9 | 0.1 | 0.413 [0.350, 0.480] | -0.170 [-0.251, -0.090] | 0.000111 (0.00122) | 0.60, 0.62 |
| hog alone | 1764 | 0.001 | 0.475 [0.408, 0.543] | -0.108 [-0.157, -0.058] | 0.000116 (0.00122) | 0.46, 0.48 |
| lbp alone | 10 | 1 | 0.610 [0.543, 0.673] | +0.027 [-0.058, +0.108] | 0.594 (1) | 0.47, 0.49 |
| all minus hsv | 1783 | 0.001 | 0.565 [0.502, 0.628] | -0.018 [-0.054, +0.018] | 0.454 (1) | 1.56, 1.60 |
| all minus moments | 1966 | 0.001 | 0.574 [0.507, 0.641] | -0.009 [-0.031, +0.009] | 0.688 (1) | 2.32, 2.41 |
| all minus hog | 211 | 0.1 | 0.641 [0.578, 0.704] | +0.058 [-0.013, +0.130] | 0.16 (1) | 2.43, 2.52 |
| all minus lbp | 1965 | 0.001 | 0.547 [0.480, 0.610] | -0.036 [-0.072, +0.000] | 0.0768 (0.615) | 2.46, 2.72 |
| HOG 16x16 cells | 535 | 0.001 | 0.610 [0.543, 0.668] | +0.027 [-0.022, +0.081] | 0.405 (1) | 2.76, 2.87 |
| HSV bins (8,3,3) | 1855 | 0.001 | 0.574 [0.507, 0.641] | -0.009 [-0.031, +0.013] | 0.688 (1) | 2.89, 2.98 |
| LBP P=16 R=2 | 1983 | 0.01 | 0.596 [0.529, 0.659] | +0.013 [-0.027, +0.054] | 0.678 (1) | 3.28, 3.39 |

Variants above the default (accuracy above the default's CI, or paired CI above 0): none.
Default reproduces step 1 (results/01_baseline/handcrafted.json): True.
