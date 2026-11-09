"""Turn each tile into a 1024-number vector with a frozen ResNet-50.

This is the expensive step, and you do it ONCE. After that, every experiment
(baseline, attention MIL, ...) trains on the saved vectors in seconds/minutes.

Why truncate after layer3? It is what CLAM does: layer3 features are more
"texture/tissue" and less "ImageNet object", and 1024-d is lighter than 2048-d.
"""
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Trunc(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.body = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    @torch.no_grad()
    def forward(self, x):                       # x: (B, 3, H, W)
        return self.pool(self.body(x)).flatten(1)   # (B, 1024)


def build_extractor(device: str = "cpu") -> tuple[nn.Module, transforms.Compose]:
    model = ResNet50Trunc().eval().to(device)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, tf

# UPGRADE IDEA (after the baseline works): swap ResNet-50/ImageNet for a
# pathology foundation model (e.g. Phikon / Phikon-v2 from Owkin, UNI, CONCH).
# Same pipeline, just a different encoder and feat_dim. Usually a big jump.
