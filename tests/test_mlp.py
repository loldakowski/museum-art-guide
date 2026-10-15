import math

import torch
import torch.nn.functional as F

from museum_guide.mlp import MLP, Linear, inverted_dropout


def test_linear_is_affine_map():
    torch.manual_seed(0)
    layer = Linear(7, 3)
    x = torch.randn(5, 7)
    assert torch.allclose(layer(x), x @ layer.weight.T + layer.bias)


def test_he_init_variance():
    torch.manual_seed(0)
    layer = Linear(2000, 1000, init="he")
    assert abs(layer.weight.std().item() - math.sqrt(2 / 2000)) < 1e-3


def test_inverted_dropout_preserves_expectation_and_is_identity_in_eval():
    torch.manual_seed(0)
    x = torch.ones(200_000)
    assert abs(inverted_dropout(x, 0.5, True).mean().item() - 1.0) < 0.01
    assert torch.equal(inverted_dropout(x, 0.5, False), x)


def test_initial_loss_is_ln_k():
    torch.manual_seed(0)
    m = MLP(300, 5)
    loss = F.cross_entropy(m(torch.randn(512, 300)), torch.randint(0, 5, (512,)))
    assert abs(loss.item() - math.log(5)) < 0.05


def test_gradients_reach_every_parameter():
    torch.manual_seed(0)
    m = MLP(20, 3, hidden=(16, 8), dropout=0.0)
    F.cross_entropy(m(torch.randn(32, 20)), torch.randint(0, 3, (32,))).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_can_overfit_a_tiny_batch():
    torch.manual_seed(0)
    x, y = torch.randn(16, 50), torch.randint(0, 4, (16,))
    m = MLP(50, 4, hidden=(64,), dropout=0.0)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
    assert (m(x).argmax(1) == y).all()


def test_zero_init_does_not_learn_hidden_weights():
    m = MLP(10, 3, hidden=(8,), init="zeros", dropout=0.0)
    F.cross_entropy(m(torch.randn(4, 10)), torch.tensor([0, 1, 2, 0])).backward()
    assert m.hidden[0].weight.grad.abs().sum() == 0  # symmetry: nothing to break it


def test_normalisation_buffers_are_saved():
    m = MLP(4, 2).set_normalization(torch.arange(4.0), torch.full((4,), 2.0))
    sd = m.state_dict()
    assert torch.equal(sd["mean"], torch.arange(4.0)) and torch.equal(sd["std"], torch.full((4,), 2.0))
