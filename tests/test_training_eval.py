import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from museum_guide.evaluation import bootstrap_ci, classification_metrics
from museum_guide.mlp import MLP
from museum_guide.baseline import make_logreg
from museum_guide.common import provenance
from museum_guide.resnet import FrozenLogReg, ResNetClassifier, make_head, save_frozen_logreg
from museum_guide.training import TrainConfig, train


def _blobs(n=300, d=10, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % k
    x = torch.randn(n, d, generator=g) + 3 * torch.nn.functional.one_hot(y, d).float()
    return x, y


def _loaders():
    x, y = _blobs()
    return (DataLoader(TensorDataset(x[:200], y[:200]), batch_size=32, shuffle=True),
            DataLoader(TensorDataset(x[200:], y[200:]), batch_size=100))


def test_training_learns_and_records_monitoring():
    torch.manual_seed(0)
    tr, va = _loaders()
    h = train(MLP(10, 3, hidden=(32,)), tr, va, TrainConfig(epochs=15, lr=1e-2), log=lambda *_: None)
    assert h["val_acc"][h["best_epoch"] - 1] > 0.9
    assert h["train_loss"][-1] < h["initial_val_loss"]
    assert set(h["grad_norm"]) == {"hidden.0", "out"}
    assert all(len(v) == len(h["epoch"]) for v in h["update_ratio"].values())
    assert "hidden0" in h["hidden_stats"]


def test_divergence_is_detected():
    torch.manual_seed(0)
    tr, va = _loaders()
    cfg = TrainConfig(epochs=20, lr=1e6, optimizer="sgd", momentum=0.0, weight_decay=0.0, patience=None)
    h = train(MLP(10, 3, hidden=(32,), init="large"), tr, va, cfg, log=lambda *_: None)
    assert h["status"] == "diverged"


def test_metrics_and_bootstrap():
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    pred = y.copy()
    pred[0] = 1
    m = classification_metrics(y, pred, ["a", "b", "c"])
    assert m["accuracy"] == 0.9 and np.array(m["confusion_matrix"]).sum() == 10
    lo, hi = bootstrap_ci(y, pred)
    assert lo <= 0.9 <= hi


def test_resnet_modes_freeze_the_right_layers():
    frozen = ResNetClassifier(5, "frozen", pretrained=False)
    tuned = ResNetClassifier(5, "finetune", pretrained=False)
    assert frozen.n_parameters(True) == 512 * 5 + 5
    assert tuned.n_parameters(True) > frozen.n_parameters(True)
    assert not any(p.requires_grad for p in tuned.net.layer3.parameters())
    tuned.train()
    assert not tuned.net.layer1[0].bn1.training      # frozen BN stays on ImageNet statistics
    assert tuned.net.layer4[0].bn1.training          # trainable BN is in train mode
    assert tuned(torch.randn(2, 3, 64, 64)).shape == (2, 5)
    assert frozen.embed(torch.randn(2, 3, 64, 64)).shape == (2, 512)


def test_standardised_head_starts_at_ln_k_and_plain_xavier_does_not():
    # ResNet embeddings are non-negative averages of ReLUs with mean ~1: plain Xavier on raw inputs gives
    # large logits (the earlier head started at 2.79 instead of ln 5)
    E = torch.distributions.Exponential(1.0).sample((500, 512))
    y = torch.arange(500) % 5
    with torch.no_grad():
        loss = {v: float(torch.nn.functional.cross_entropy(make_head(512, 5, v, train_embeddings=E)(E), y))
                for v in ("plain_xavier", "small_init", "standardised")}
    assert abs(loss["standardised"] - math.log(5)) < 0.05
    assert loss["plain_xavier"] > math.log(5) + 0.5
    head = ResNetClassifier(5, "frozen", pretrained=False).net.fc
    assert torch.equal(head.mean, torch.zeros(512))  # standardisation buffers ship with the model


def test_frozen_logreg_file_round_trip(fake_data, other_fake_data, tmp_path):
    torch.manual_seed(0)
    model = ResNetClassifier(5, "frozen", pretrained=False).eval()
    x = torch.randn(10, 3, 32, 32)
    E = model.embed(x).numpy()
    logreg = make_logreg(1.0).fit(E, np.arange(10) % 5)
    f = tmp_path / "resnet18_frozen_logreg.joblib"
    save_frozen_logreg(f, model, logreg, list("abcde"), 32, provenance(fake_data))
    loaded = FrozenLogReg.load(f, fake_data)
    np.testing.assert_allclose(loaded.predict_proba(x), logreg.predict_proba(E), rtol=1e-5, atol=1e-6)
    assert loaded.classes == list("abcde")
    with pytest.raises(ValueError, match="does not match"):
        FrozenLogReg.load(f, other_fake_data)
