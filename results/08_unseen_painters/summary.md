# Cost of unseen painters (data/wikiart_named5, 5-fold cross-validation)

Same models and hyper-parameters as the main experiment. Random split: test painters also appear in training. Painter-disjoint split: every painter is tested once, never seen in training. Pooled = all test images of the 5 folds (each image tested once).

| model | random split: pooled acc | folds mean ± std | painter-disjoint: pooled acc | folds mean ± std | cost of unseen painters (points, pooled) |
|---|---|---|---|---|---|
| logreg_pixels32 | 0.333 | 0.333 ± 0.019 | 0.244 | 0.244 ± 0.024 | +8.9 |
| logreg_handcrafted | 0.369 | 0.369 ± 0.014 | 0.291 | 0.290 ± 0.041 | +7.8 |
| mlp_pixels32 | 0.312 | 0.312 ± 0.025 | 0.247 | 0.246 ± 0.028 | +6.5 |
| mlp_handcrafted | 0.344 | 0.344 ± 0.017 | 0.259 | 0.258 ± 0.029 | +8.5 |
| resnet18_frozen_linear_head | 0.519 | 0.519 ± 0.032 | 0.333 | 0.333 ± 0.062 | +18.7 |
| resnet18_frozen_logreg | 0.523 | 0.523 ± 0.042 | 0.343 | 0.343 ± 0.066 | +18.0 |
| resnet18_finetune | 0.544 | 0.544 ± 0.033 | 0.344 | 0.344 ± 0.071 | +20.0 |

## Accuracy per fold

| model | split | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|---|
| logreg_pixels32 | random | 0.314 | 0.326 | 0.320 | 0.360 | 0.343 |
| logreg_pixels32 | artist | 0.264 | 0.261 | 0.232 | 0.255 | 0.207 |
| logreg_handcrafted | random | 0.388 | 0.352 | 0.377 | 0.360 | 0.367 |
| logreg_handcrafted | artist | 0.268 | 0.293 | 0.294 | 0.353 | 0.244 |
| mlp_pixels32 | random | 0.288 | 0.322 | 0.293 | 0.350 | 0.307 |
| mlp_pixels32 | artist | 0.237 | 0.270 | 0.201 | 0.265 | 0.258 |
| mlp_handcrafted | random | 0.334 | 0.329 | 0.343 | 0.340 | 0.373 |
| mlp_handcrafted | artist | 0.241 | 0.303 | 0.239 | 0.271 | 0.237 |
| resnet18_frozen_linear_head | random | 0.565 | 0.522 | 0.477 | 0.527 | 0.507 |
| resnet18_frozen_linear_head | artist | 0.414 | 0.349 | 0.311 | 0.346 | 0.244 |
| resnet18_frozen_logreg | random | 0.555 | 0.525 | 0.450 | 0.540 | 0.543 |
| resnet18_frozen_logreg | artist | 0.447 | 0.345 | 0.324 | 0.333 | 0.264 |
| resnet18_finetune | random | 0.599 | 0.508 | 0.537 | 0.543 | 0.533 |
| resnet18_finetune | artist | 0.431 | 0.339 | 0.358 | 0.359 | 0.234 |
