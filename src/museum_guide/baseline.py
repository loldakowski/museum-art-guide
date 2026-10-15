"""Classical baseline: multinomial logistic regression, regularisation chosen on the validation set."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def make_logreg(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def fit_logreg(X_train, y_train, X_val, y_val, Cs=C_GRID):
    """Fit one model per C, keep the best on validation accuracy. Returns (model, {C: val_acc})."""
    scores, best, best_acc = {}, None, -1.0
    for C in Cs:
        m = make_logreg(C).fit(X_train, y_train)
        acc = float(np.mean(m.predict(X_val) == y_val))
        scores[C] = acc
        if acc > best_acc:
            best, best_acc = m, acc
    return best, scores
