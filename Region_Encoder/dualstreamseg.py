# models/dualstreamseg.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights


# =========================
# 1) ENCODERS
# =========================
class RegionEncoderResNet50(nn.Module):
    """Outputs multi-scale features: (1/4, 1/8, 1/16)"""
    def __init__(self, pretrained=True):
        super().__init__()
        w = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=w)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # 256
        self.layer2 = base.layer2   # 512
        self.layer3 = base.layer3   # 1024

    def forward(self, x):
        x = self.stem(x)      # 1/4
        f1 = self.layer1(x)   # 1/4
        f2 = self.layer2(f1)  # 1/8
        f3 = self.layer3(f2)  # 1/16
        return f1, f2, f3


class BoundaryEncoderResNet18(nn.Module):
    """Outputs multi-scale features: (1/4, 1/8, 1/16)"""
    def __init__(self, pretrained_imagenet=True):
        super().__init__()
        w = ResNet18_Weights.DEFAULT if pretrained_imagenet else None
        base = resnet18(weights=w)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # 64
        self.layer2 = base.layer2   # 128
        self.layer3 = base.layer3   # 256

    def forward(self, x):
        x = self.stem(x)
        b1 = self.layer1(x)   # 1/4
        b2 = self.layer2(b1)  # 1/8
        b3 = self.layer3(b2)  # 1/16
        return b1, b2, b3

    def load_boundary_pretrain(self, ckpt_path: str):
        # checkpoint may contain extra keys -> strict=False
        sd = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print("[BoundaryEncoder] loaded:", ckpt_path)
        print("  missing:", len(missing), "| unexpected:", len(unexpected))


# =========================
# 2) FUSION
# =========================
class BoundaryGuidedFuse(nn.Module):
    """Fr, Fb -> project -> boundary gate -> Fr*(1+gate)"""
    def __init__(self, c_r, c_b, c_out):
        super().__init__()
        self.r = nn.Sequential(
            nn.Conv2d(c_r, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
        self.b = nn.Sequential(
            nn.Conv2d(c_b, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Conv2d(c_out, c_out, 1)

    def forward(self, Fr, Fb):
        r = self.r(Fr)
        b = self.b(Fb)
        g = torch.sigmoid(self.gate(b))
        return r * (1.0 + g)


# =========================
# 3) DECODER BLOCKS
# =========================
class UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in + c_skip, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


# =========================
# 4) FULL MODEL
# =========================
class DualStreamSeg(nn.Module):
    def __init__(self, boundary_ckpt=None, freeze_boundary=True, freeze_region=True, c_f=128):
        super().__init__()
        self.region = RegionEncoderResNet50(pretrained=True)
        self.boundary = BoundaryEncoderResNet18(pretrained_imagenet=True)

        if boundary_ckpt is not None:
            self.boundary.load_boundary_pretrain(boundary_ckpt)

        self.fuse1 = BoundaryGuidedFuse(256, 64,  c_f)   # 1/4
        self.fuse2 = BoundaryGuidedFuse(512, 128, c_f)   # 1/8
        self.fuse3 = BoundaryGuidedFuse(1024, 256, c_f)  # 1/16

        self.up32 = UpBlock(c_f, c_f, c_f)  # 1/16 -> 1/8
        self.up21 = UpBlock(c_f, c_f, c_f)  # 1/8  -> 1/4

        self.head = nn.Sequential(
            nn.Conv2d(c_f, c_f, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_f),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_f, 1, 1)
        )

        if freeze_boundary:
            for p in self.boundary.parameters():
                p.requires_grad = False
        if freeze_region:
            for p in self.region.parameters():
                p.requires_grad = False

    def forward(self, x):
        r1, r2, r3 = self.region(x)
        b1, b2, b3 = self.boundary(x)

        f1 = self.fuse1(r1, b1)
        f2 = self.fuse2(r2, b2)
        f3 = self.fuse3(r3, b3)

        x = self.up32(f3, f2)
        x = self.up21(x,  f1)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return self.head(x)
