"""Pretrained ResNet-18 for style classification (deliverable 3), two transfer-learning regimes.

mode='frozen'   : the ImageNet backbone is a fixed feature extractor (512-d embedding after global
                  average pooling); only a new linear head is trained. Cheap: embeddings are computed
                  once, the head trains in seconds on CPU.
mode='finetune' : the last residual stage (layer4) is also trained, with a 10x smaller learning rate
                  than the head so that the pretrained filters are adapted, not destroyed.
Frozen BatchNorm layers stay in eval mode (their running statistics come from ImageNet).

The head is our MLP with no hidden layer (mlp.py): input standardisation (buffers, fitted on the training
embeddings in frozen mode) and a x0.1 Xavier output layer, so that the initial loss is ~ln K, as for the MLP.
make_head() also builds the earlier head (plain Xavier, raw embeddings) for the ablation in step 4.

The recommended kiosk model is the frozen backbone + a logistic regression on its embeddings:
save_frozen_logreg / FrozenLogReg store and run it as one self-contained file.
"""
from __future__ import annotations

import joblib
import torch
import torch.nn as nn

from .common import check_checkpoint
from .mlp import MLP

HEAD_VARIANTS = ("plain_xavier", "small_init", "standardised")


class ResNetClassifier(nn.Module):
    def __init__(self, n_classes: int, mode: str = "finetune", pretrained: bool = True,
                 backbone_lr_factor: float = 0.1):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.embed_dim = self.net.fc.in_features
        self.net.fc = make_head(self.embed_dim, n_classes)  # our own primitive layers as the head
        self.mode, self.backbone_lr_factor = mode, backbone_lr_factor
        for p in self.net.parameters():
            p.requires_grad = False
        for p in self.net.fc.parameters():
            p.requires_grad = True
        if mode == "finetune":
            for p in self.net.layer4.parameters():
                p.requires_grad = True
        elif mode != "frozen":
            raise ValueError(mode)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.net.modules():  # keep frozen BatchNorms on their ImageNet statistics
            if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                m.eval()
        return self

    def param_groups(self, lr: float):
        head = list(self.net.fc.parameters())
        body = [p for n, p in self.net.named_parameters() if p.requires_grad and not n.startswith("fc.")]
        groups = [{"params": head, "lr": lr}]
        if body:
            groups.append({"params": body, "lr": lr * self.backbone_lr_factor})
        return groups

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def embed(self, x):
        """512-d embedding (everything except the classification head)."""
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        return torch.flatten(n.avgpool(x), 1)

    def n_parameters(self, trainable_only=False):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


@torch.no_grad()
def extract_embeddings(model: ResNetClassifier, loader, device="cpu"):
    model.eval().to(device)
    E, Y = [], []
    for xb, yb in loader:
        E.append(model.embed(xb.to(device)).cpu())
        Y.append(yb)
    return torch.cat(E), torch.cat(Y)


def make_head(dim: int, n_classes: int, variant: str = "standardised", train_embeddings=None) -> MLP:
    """Linear classification head = MLP without hidden layer.

    'standardised' : x0.1 Xavier output layer + standardisation fitted on train_embeddings (if given)
    'small_init'   : x0.1 Xavier output layer, raw embeddings
    'plain_xavier' : plain Xavier, raw embeddings (the head used before; initial loss 2.79 instead of ln 5)
    """
    head = MLP(dim, n_classes, hidden=(), dropout=0.0)
    if variant == "plain_xavier":
        head.out.reset_parameters("xavier")
    elif variant not in HEAD_VARIANTS:
        raise ValueError(variant)
    if variant == "standardised" and train_embeddings is not None:
        head.set_normalization(train_embeddings.mean(0), train_embeddings.std(0))
    return head


def save_frozen_logreg(path, model: ResNetClassifier, logreg, classes, img_size: int, prov: dict) -> None:
    """The deployable kiosk model in one file: backbone weights (head left out) + scaler/logistic regression."""
    backbone = {k: v for k, v in model.state_dict().items() if not k.startswith("net.fc.")}
    joblib.dump({"backbone_state_dict": backbone, "logreg": logreg, "classes": list(classes),
                 "img_size": img_size, "provenance": prov}, path)


class FrozenLogReg:
    """Photo tensor -> class probabilities with a file written by save_frozen_logreg (no download needed).
    The file must come from the dataset in data_dir (check_checkpoint), as for every other checkpoint."""

    def __init__(self, model: ResNetClassifier, logreg, classes, img_size: int):
        self.model, self.logreg, self.classes, self.img_size = model.eval(), logreg, list(classes), img_size

    @classmethod
    def load(cls, path, data_dir) -> "FrozenLogReg":
        ck = joblib.load(path)
        check_checkpoint(ck, data_dir, path)
        model = ResNetClassifier(len(ck["classes"]), "frozen", pretrained=False)
        missing, unexpected = model.load_state_dict(ck["backbone_state_dict"], strict=False)
        if unexpected or any(not k.startswith("net.fc.") for k in missing):
            raise ValueError(f"{path}: backbone weights do not match ResNet-18 ({missing[:3]}, {unexpected[:3]})")
        return cls(model, ck["logreg"], ck["classes"], ck["img_size"])

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor):
        return self.logreg.predict_proba(self.model.embed(x).numpy())
