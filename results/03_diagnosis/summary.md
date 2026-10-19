# MLP failure diagnosis

| scenario | status | initial_loss | final_train_acc | best_val_acc | test_acc | grad_first_over_out | mean_update_ratio | dead_or_saturated_end | expected symptom |
|---|---|---|---|---|---|---|---|---|---|
| reference | early_stopped | 1.629 | 0.791 | 0.426 | 0.390 | 1.02e-01 | 1.01e-02 | 0.031 | healthy: val loss decreases, then early stopping |
| overfit_one_batch | ok | 1.630 | 1.000 | 1.000 | 0.266 | 2.03e-01 | 2.66e-03 | 0.006 | train acc reaches ~100 % on 32 images (the code can learn) |
| lr_too_high | early_stopped | 1.629 | 0.208 | 0.206 | 0.187 | 2.26e-02 | 4.16e-02 | 0.500 | loss explodes or stays at chance, many dead ReLUs |
| lr_too_low | ok | 1.629 | 0.317 | 0.314 | 0.307 | 3.34e-01 | 1.05e-05 | 0.000 | loss stays near ln(K), update/weight ratio << 1e-4 |
| init_zeros | early_stopped | 1.609 | 0.201 | 0.202 | 0.187 | 0.00e+00 | 0.00e+00 | 1.000 | symmetry: loss stuck at ln(K), zero gradients in hidden layers |
| init_large | early_stopped | 9581.822 | 0.742 | 0.350 | 0.320 | 9.89e-01 | 1.04e-04 | 0.000 | initial loss >> ln(K), unstable training |
| deep_sigmoid | early_stopped | 1.609 | 0.191 | 0.220 | 0.224 | 7.46e-05 | 4.92e-02 | 0.000 | vanishing gradients: first-layer grad norm << last-layer grad norm |
| no_normalisation | early_stopped | 1.613 | 0.604 | 0.408 | 0.407 | 8.53e-02 | 4.49e-03 | 0.462 | slower or worse convergence than the reference |
| no_regularisation | ok | 1.629 | 0.924 | 0.372 | 0.402 | 1.04e-01 | 4.40e-03 | 0.066 | train acc ~100 %, val loss rises after its minimum (overfitting) |
| shuffled_labels | ok | 1.629 | 1.000 | 0.265 | 0.183 | 1.06e-01 | 1.93e-03 | 0.029 | train acc rises, val/test acc ~ chance (memorisation, no leakage) |
| random_split | early_stopped | 1.636 | 0.837 | 0.413 | 0.447 | 1.08e-01 | 1.02e-02 | 0.018 | test acc higher than with the artist split (leakage through painters) |
