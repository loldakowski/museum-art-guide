# Results summary (test set, artist-grouped split)

| model | test acc [95% CI] | macro-F1 | latency (ms, 1 thread) | size (MB) | params | train time (s) | note |
|---|---|---|---|---|---|---|---|
| LogReg - pixels32 | 0.352 [0.288-0.415] | 0.343 | 1.9 | 0.2 | 15,365 | 40 |  |
| LogReg - handcrafted | 0.627 [0.568-0.691] | 0.623 | 16.7 | 0.1 | 9,880 | 4 |  |
| MLP - pixels32 | 0.347 [0.288-0.411] | 0.318 | 1.3 | 6.9 | 1,705,989 | 1 | 3 seeds: 0.357 +/- 0.011 |
| MLP - handcrafted | 0.589 [0.530-0.653] | 0.584 | - | - | - | 1 | 3 seeds: 0.576 +/- 0.015 |
| ResNet-18 frozen + linear head | 0.699 [0.640-0.754] | 0.696 | 23.1 | 44.8 | 11,179,077 | 1 |  |
| ResNet-18 frozen + LogReg | 0.763 [0.708-0.814] | 0.762 | 23.1 | 44.8 | 11,179,077 | - |  |
| ResNet-18 fine-tuned | 0.771 [0.716-0.826] | 0.767 | 23.1 | 44.8 | 11,179,077 | 63 |  |

Latency = median end-to-end time for one photo on CPU (see 05_benchmark).
