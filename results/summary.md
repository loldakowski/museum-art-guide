# Results summary (test set, artist-grouped split)

| model | test acc [95% CI] (saved model) | mean ± std over seeds | macro-F1 | latency (ms, 1 thread) | latency, draft decoding (ms) | size (MB) | params | train time (s) | note |
|---|---|---|---|---|---|---|---|---|---|
| LogReg - pixels32 | 0.349 [0.286-0.411] | 1 seed | 0.331 | 269.4 | 42.2 | 0.1 | 15,365 | 28 |  |
| LogReg - handcrafted | 0.610 [0.548-0.668] | 1 seed | 0.607 | 285.1 | 58.2 | 0.1 | 9,880 | 3 |  |
| MLP - pixels32 | 0.390 [0.328-0.448] | 0.385 ± 0.010 (3 seeds) | 0.375 | 255.4 | 41.6 | 6.9 | 1,705,989 | 1 |  |
| MLP - handcrafted | 0.593 [0.531-0.660] | 0.571 ± 0.050 (3 seeds) | 0.592 | 274.0 | 57.8 | 10.7 | 2,680,837 | 1 |  |
| ResNet-18 frozen + linear head | 0.763 [0.710-0.817] | 1 seed | 0.760 | 247.9 | 63.8 | 44.8 | 11,179,077 | 1 |  |
| ResNet-18 frozen + LogReg | 0.759 [0.705-0.813] | 1 seed | 0.757 | 259.1 | 65.3 | 44.8 | 11,179,077 | - |  |
| ResNet-18 fine-tuned | 0.813 [0.763-0.863] | 1 seed | 0.812 | 247.9 | 63.8 | 44.8 | 11,179,077 | 47 |  |

Latency = median end-to-end time for one phone photo on CPU, decoding included (see 05_benchmark). Draft decoding = im.draft() + ImageOps.fit, the preprocessing that built the training images.

## Paired comparisons: ResNet-18 frozen + LogReg vs each model (same 241 test images)

Difference = accuracy of the recommended model minus the other model, 95 % CI from a paired bootstrap over test images. McNemar: exact test on the images where exactly one of the two models is right; Holm = p-value adjusted for the 6 comparisons. MLP rows: the saved model (seed 0).

| other model | its acc | difference [95% CI] | only recommended right | only other right | McNemar p | Holm p |
|---|---|---|---|---|---|---|
| LogReg - pixels32 | 0.349 | +0.411 [+0.340, +0.490] | 112 | 13 | 8.12e-21 | 4.87e-20 |
| LogReg - handcrafted | 0.610 | +0.149 [+0.079, +0.224] | 61 | 25 | 0.00013 | 0.000391 |
| MLP - pixels32 | 0.390 | +0.369 [+0.295, +0.448] | 105 | 16 | 3.16e-17 | 1.58e-16 |
| MLP - handcrafted | 0.593 | +0.166 [+0.091, +0.245] | 69 | 29 | 6.57e-05 | 0.000263 |
| ResNet-18 frozen + linear head | 0.763 | -0.004 [-0.029, +0.021] | 4 | 5 | 1 | 1 |
| ResNet-18 fine-tuned | 0.813 | -0.054 [-0.104, -0.004] | 14 | 27 | 0.0596 | 0.119 |

## Free-text search (see 07_search/summary.md)

| system | P@10 metadata queries (TF-IDF: upper bound) | P@10 content queries (judged) | latency per query (ms, 1 thread) | size (MB) |
|---|---|---|---|---|
| CLIP ViT-B-32-quickgelu/openai | 0.833 | 0.550 | 115.7 | 605 + index 3.1 |
| TF-IDF on metadata | 0.867 | 0.025 | 1.09 | 0.43 |
| random | - | 0.005 | - | - |

Code that produced these numbers (git commit):
- `747ac0eee787410e85c2e1358c8093c00efdbc72`: 00_explore/duplicates_report.json, 00_explore/eda.json, 01_baseline/handcrafted.json, 01_baseline/handcrafted_randomsplit.json, 01_baseline/pixels32.json, 01_baseline/pixels32_randomsplit.json, 02_mlp/handcrafted/summary.json, 02_mlp/pixels32/summary.json, 03_diagnosis/summary.json, 04_resnet/finetune/summary.json, 04_resnet/frozen/summary.json, 05_benchmark/benchmark.json
