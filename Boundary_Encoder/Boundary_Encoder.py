import os, glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt


# -------------------------
# 1) Dataset
# -------------------------
def mask_to_boundary(mask01, k=3):
    m = (mask01 > 0).astype(np.uint8)
    kernel = np.ones((k, k), np.uint8)
    dil = cv2.dilate(m, kernel, iterations=1)
    ero = cv2.erode(m, kernel, iterations=1)
    bd = (dil - ero)
    bd = (bd > 0).astype(np.float32)
    return bd

class CellsDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=256):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_dir = mask_dir
        self.size = size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, base + ".png")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")

        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        mask01 = (mask > 0).astype(np.float32)
        boundary = mask_to_boundary(mask01, k=3)

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)          # [3,H,W]
        boundary = torch.from_numpy(boundary).unsqueeze(0)    # [1,H,W]
        return img, boundary, base


# -------------------------
# 2) Model: ResNet18 encoder + boundary head
# -------------------------
class ResNet18Boundary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

        # lightweight decoder to full res
        self.up2 = nn.Conv2d(256, 128, 1)
        self.up1 = nn.Conv2d(128, 64, 1)

        # stronger boundary head than 1x1 only
        self.bnd_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        x = self.stem(x)      # 1/4
        x = self.layer1(x)    # 1/4
        x = self.layer2(x)    # 1/8
        x = self.layer3(x)    # 1/16

        x = F.interpolate(self.up2(x), scale_factor=2, mode="bilinear", align_corners=False)  # 1/8
        x = F.interpolate(self.up1(x), scale_factor=2, mode="bilinear", align_corners=False)  # 1/4
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)            # 1/1

        bnd_logits = self.bnd_head(x)  # raw logits
        return bnd_logits


# -------------------------
# 3) Loss: weighted BCE for sparse boundary
# -------------------------
def boundary_loss(bnd_logits, bnd_gt):
    # bnd_gt is {0,1}
    pos = bnd_gt.sum().clamp(min=1.0)
    neg = (bnd_gt.numel() - bnd_gt.sum()).clamp(min=1.0)
    pos_weight = (neg / pos).clamp(max=50.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(bnd_logits, bnd_gt)


# -------------------------
# 4) Visualize
# -------------------------
def save_preview(img, bnd_gt, bnd_pred, out_path):
    img = img.permute(1,2,0).cpu().numpy()
    bnd_gt = bnd_gt[0].cpu().numpy()
    bnd_pred = bnd_pred[0].cpu().numpy()

    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1); plt.title("Image"); plt.imshow(img); plt.axis("off")
    plt.subplot(1,3,2); plt.title("GT Boundary"); plt.imshow(bnd_gt, cmap="gray"); plt.axis("off")
    plt.subplot(1,3,3); plt.title("Pred Boundary"); plt.imshow(bnd_pred, cmap="hot"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    data_root = "data/BCCD Dataset with mask/test"
    img_dir = os.path.join(data_root, "original")
    mask_dir = os.path.join(data_root, "mask")
    out_dir = "previews_boundary_only"
    os.makedirs(out_dir, exist_ok=True)

    ds = CellsDataset(img_dir, mask_dir, size=256)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet18Boundary(pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 100
    model.train()

    for ep in range(1, epochs + 1):
        total = 0.0
        for img, bnd_gt, name in dl:
            img = img.to(device)
            bnd_gt = bnd_gt.to(device)

            bnd_logits = model(img)
            loss = boundary_loss(bnd_logits, bnd_gt)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        print(f"Epoch {ep}/{epochs} | boundary loss={total/len(dl):.4f}")

        # preview
        model.eval()
        with torch.no_grad():
            img, bnd_gt, name = next(iter(dl))
            img = img.to(device)
            bnd_logits = model(img)
            bnd_pred = torch.sigmoid(bnd_logits).cpu()

            save_preview(img[0].cpu(), bnd_gt[0], bnd_pred[0],
                         os.path.join(out_dir, f"epoch_{ep}.png"))
        model.train()

    print("Done. Check previews_boundary_only/epoch_*.png")

if __name__ == "__main__":
    main()
