# MLP failure diagnosis

| scenario | status | initial_loss | final_train_acc | best_val_acc | test_acc | grad_first_over_out | mean_update_ratio | dead_or_saturated_end | expected symptom |
|---|---|---|---|---|---|---|---|---|---|
| reference | early_stopped | 1.626 | 0.829 | 0.437 | 0.339 | 1.02e-01 | 1.04e-02 | 0.014 | healthy: val loss decreases, then early stopping |
| overfit_one_batch | ok | 1.638 | 1.000 | 1.000 | 0.301 | 1.97e-01 | 2.75e-03 | 0.012 | train acc reaches ~100 % on 32 images (the code can learn) |
| lr_too_high | early_stopped | 1.626 | 0.193 | 0.205 | 0.208 | 1.80e-02 | 5.02e-02 | 0.498 | loss explodes or stays at chance, many dead ReLUs |
| lr_too_low | ok | 1.626 | 0.312 | 0.221 | 0.339 | 3.25e-01 | 1.18e-05 | 0.000 | loss stays near ln(K), update/weight ratio << 1e-4 |
| init_zeros | early_stopped | 1.609 | 0.210 | 0.171 | 0.191 | 0.00e+00 | 0.00e+00 | 1.000 | symmetry: loss stuck at ln(K), zero gradients in hidden layers |
| init_large | ok | 9316.995 | 0.681 | 0.346 | 0.335 | 9.81e-01 | 1.05e-04 | 0.002 | initial loss >> ln(K), unstable training |
| deep_sigmoid | early_stopped | 1.616 | 0.206 | 0.171 | 0.191 | 1.36e-04 | 1.83e-02 | 0.000 | vanishing gradients: first-layer grad norm << last-layer grad norm |
| no_normalisation | early_stopped | 1.605 | 0.535 | 0.422 | 0.369 | 8.16e-02 | 5.46e-03 | 0.468 | slower or worse convergence than the reference |
| no_regularisation | ok | 1.626 | 0.994 | 0.403 | 0.356 | 1.07e-01 | 5.07e-03 | 0.016 | train acc ~100 %, val loss rises after its minimum (overfitting) |
| shuffled_labels | ok | 1.626 | 1.000 | 0.190 | 0.216 | 1.18e-01 | 1.95e-03 | 0.014 | train acc rises, val/test acc ~ chance (memorisation, no leakage) |
| random_split | early_stopped | 1.628 | 0.821 | 0.385 | 0.387 | 1.04e-01 | 1.04e-02 | 0.021 | test acc higher than with the artist split (leakage through painters) |
