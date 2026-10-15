"""Pretrained ResNet-18 for style classification (deliverable 3), two transfer-learning regimes.

mode='frozen'   : the ImageNet backbone is a fixed feature extractor (512-d embedding after global
                  average pooling); only a new linear head is trained. Cheap: embeddings are computed
                  once, the head trains in seconds on CPU.
mode='finetune' : the last residual stage (layer4) is also trained, with a 10x smaller learning rate
                  than the head so that the pretrained filters are adapted, not destroyed.
Frozen BatchNorm layers stay in eval mode (their running statistics come from ImageNet).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import Linear


class ResNetClassifier(nn.Module):
    def __init__(self, n_classes: int, mode: str = "finetune", pretrained: bool = True,
                 backbone_lr_factor: float = 0.1):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.embed_dim = self.net.fc.in_features
        self.net.fc = Linear(self.embed_dim, n_classes, init="xavier")  # our own primitive layer as the head
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


class LinearHead(nn.Module):
    """The classification head alone, trained on cached embeddings (mode='frozen')."""

    def __init__(self, dim, n_classes):
        super().__init__()
        self.fc = Linear(dim, n_classes, init="xavier")

    def forward(self, x):
        return self.fc(x)
