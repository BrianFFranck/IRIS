"""Turn each tile into a feature vector with a frozen encoder.

  resnet50  ImageNet ResNet-50 truncated after layer3      -> 1024 numbers (the original baseline)
  hibou-b   HistAI Hibou-B, pathology ViT-B/14, Apache 2.0  ->  768 numbers
  midnight  kaiko.ai Midnight-12k, pathology ViT-g/14, MIT  -> 3072 numbers

The two pathology encoders come from Hugging Face and are downloaded the first time
they are used (Midnight is several GB). Hibou-B is "gated": it needs a Hugging Face
account that accepted its terms, and a token in the HF_TOKEN environment variable.
Both licences allow commercial use (checked 2026-11-17, see experiments.md).
"""
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

FEAT_DIMS = {"resnet50": 1024, "hibou-b": 768, "midnight": 3072}


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


def hibou_vector(out):
    """Hibou-B: its own summary vector of the tile (the normalised CLS token)."""
    return out.pooler_output


def midnight_vector(out):
    """Midnight: the CLS token joined with the average of the patch tokens,
    as its model card recommends for classification (1536 + 1536 = 3072)."""
    tokens = out.last_hidden_state
    return torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=-1)


class HFEncoder(nn.Module):
    """A pathology ViT from Hugging Face; `to_vector` turns its outputs into one vector per tile."""

    def __init__(self, name, to_vector, model_class="AutoModel"):
        super().__init__()
        import transformers                    # only needed for these encoders
        self.model = getattr(transformers, model_class).from_pretrained(name)
        self.to_vector = to_vector

    @torch.no_grad()
    def forward(self, x):
        # half precision on the GPU: about twice as fast, and the vectors are saved as float16 anyway
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
            out = self.model(pixel_values=x)
        return self.to_vector(out).float()


def build_extractor(device: str = "cpu", encoder: str = "resnet50") -> tuple[nn.Module, transforms.Compose]:
    if encoder == "resnet50":
        model = ResNet50Trunc()
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)     # ImageNet colours
    elif encoder == "hibou-b":
        # Hibou-B ships its own model code, written for transformers 4 (it breaks on 5).
        # It is a standard DINOv2 with 4 register tokens, so the class built into
        # transformers loads all its weights; checked 2026-11-17: same output as
        # Hibou's own code under transformers 4.57 (max difference 3e-6).
        model = HFEncoder("histai/hibou-b", hibou_vector, model_class="Dinov2WithRegistersModel")
        mean, std = (0.7068, 0.5755, 0.722), (0.195, 0.2316, 0.1816)  # Hibou's preprocessor_config.json
    elif encoder == "midnight":
        model = HFEncoder("kaiko-ai/midnight", midnight_vector)
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)                 # from Midnight's model card
    else:
        raise ValueError(f"unknown encoder {encoder!r}; choose from {list(FEAT_DIMS)}")
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, antialias=True),     # 256 px tiles -> 224 px, the size all three were trained on
        transforms.Normalize(mean=mean, std=std),
    ])
    return model.eval().to(device), tf
