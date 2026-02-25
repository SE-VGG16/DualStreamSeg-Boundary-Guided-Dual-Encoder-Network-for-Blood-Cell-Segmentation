# train.py
import os, glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from models import DualStreamSeg


# =========================
# Metrics
# =========================
def compute_metrics(prob, gt, thr=0.5, eps=1e-6):
    pred = (prob > thr).float()
    tp = (pred * gt).sum(dim=(1,2,3))
    fp = (pred * (1 - gt)).sum(dim=(1,2,3))
    fn = ((1 - pred) * gt).sum(dim=(1,2,3))
    tn = ((1 - pred) * (1 - gt)).sum(dim=(1,2,3))

    dice = (2*tp + eps) / (2*tp + fp + fn + eps)
    iou  = (tp + eps) / (tp + fp + fn + eps)
    prec = (tp + eps) / (tp + fp + eps)
    rec  = (tp + eps) / (tp + fn + eps)
    acc  = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return dice.mean().item(), iou.mean().item(), prec.mean().item(), rec.mean().item(), acc.mean().item()


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    dices, ious, precs, recs, accs = [], [], [], [], []
    for img, mask, _ in loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        prob = torch.sigmoid(logits)
        d, i, p, r, a = compute_metrics(prob, mask, thr=thr)
        dices.append(d); ious.append(i); precs.append(p); recs.append(r); accs.append(a)
    return float(np.mean(dices)), float(np.mean(ious)), float(np.mean(precs)), float(np.mean(recs)), float(np.mean(accs))


# =========================
# Dataset
# =========================
class CellsSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=256):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_dir = mask_dir
        self.size = size
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found: {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, name + ".png")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        return img, mask, name


# =========================
# Loss
# =========================
def dice_loss_from_logits(logits, targets, eps=1e-6):
    p = torch.sigmoid(logits).view(logits.size(0), -1)
    t = targets.view(targets.size(0), -1)
    inter = (p * t).sum(1)
    union = p.sum(1) + t.sum(1)
    return 1 - ((2*inter + eps) / (union + eps)).mean()

def seg_loss(logits, targets):
    bce = nn.BCEWithLogitsLoss()(logits, targets)
    dsc = dice_loss_from_logits(logits, targets)
    return 0.5*bce + 0.5*dsc


# =========================
# Preview
# =========================
@torch.no_grad()
def save_preview(img, gt, pred_prob, path, thr=0.5):
    img = img.permute(1,2,0).cpu().numpy()
    gt = (gt[0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    prob = pred_prob[0].cpu().numpy()
    pred = (prob > thr).astype(np.uint8) * 255

    plt.figure(figsize=(12,3))
    plt.subplot(1,4,1); plt.imshow(img); plt.title("Image"); plt.axis("off")
    plt.subplot(1,4,2); plt.imshow(gt, cmap="gray"); plt.title("GT Mask"); plt.axis("off")
    plt.subplot(1,4,3); plt.imshow(pred, cmap="gray"); plt.title(f"Pred thr={thr}"); plt.axis("off")
    plt.subplot(1,4,4); plt.imshow(prob, cmap="hot"); plt.title("Pred Prob"); plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


# =========================
# Train main
# =========================
def main():
    TRAIN_ROOT = r"data/BCCD Dataset with mask/train"
    TEST_ROOT  = r"data/BCCD Dataset with mask/test"
    BOUNDARY_CKPT = r"boundary_results/boundary_resnet18_best.pth"
    out_dir = "full_dualstream_runs"
    os.makedirs(out_dir, exist_ok=True)

    img_dir = os.path.join(TRAIN_ROOT, "original")
    mask_dir = os.path.join(TRAIN_ROOT, "mask")
    test_img_dir = os.path.join(TEST_ROOT, "original")
    test_mask_dir = os.path.join(TEST_ROOT, "mask")

    ds = CellsSegDataset(img_dir, mask_dir, size=256)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    test_ds = CellsSegDataset(test_img_dir, test_mask_dir, size=256)
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = DualStreamSeg(
        boundary_ckpt=BOUNDARY_CKPT,
        freeze_boundary=True,
        freeze_region=True,
        c_f=128
    ).to(device)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    EPOCHS = 75
    patience = 10
    best_dice = 0.0
    no_improve = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        total = 0.0

        for img, mask, _ in dl:
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            loss = seg_loss(logits, mask)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / max(1, len(dl))
        dice, iou, prec, rec, acc = evaluate(model, test_dl, device, thr=0.5)

        print(f"Epoch {ep}/{EPOCHS} | TrainLoss={avg:.4f} | Dice={dice:.4f} IoU={iou:.4f} Prec={prec:.4f} Rec={rec:.4f} Acc={acc:.4f}")

        if dice > best_dice:
            best_dice = dice
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_full_model.pth"))
            print("✔ Best model saved")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f" Early stopping at epoch {ep}")
                break

        # Unfreeze at epoch 5
        if ep == 5:
            for p in model.boundary.parameters():
                p.requires_grad = True
            for p in model.region.parameters():
                p.requires_grad = True

            opt = torch.optim.Adam([
                {"params": model.fuse1.parameters(), "lr": 1e-4},
                {"params": model.fuse2.parameters(), "lr": 1e-4},
                {"params": model.fuse3.parameters(), "lr": 1e-4},
                {"params": model.up32.parameters(),  "lr": 1e-4},
                {"params": model.up21.parameters(),  "lr": 1e-4},
                {"params": model.head.parameters(),  "lr": 1e-4},
                {"params": model.region.parameters(),   "lr": 1e-5},
                {"params": model.boundary.parameters(), "lr": 1e-5},
            ])
            print("Unfroze encoders ✔")

        # preview (random sample so it changes)
        idx = np.random.randint(len(test_ds))
        img, mask, _ = test_ds[idx]
        img_b = img.unsqueeze(0).to(device)
        prob = torch.sigmoid(model(img_b)).cpu()
        save_preview(img, mask, prob[0], os.path.join(out_dir, "test_vis", f"ep{ep:03d}.png"), thr=0.5)

    print("Done. Output:", out_dir)


if __name__ == "__main__":
    main()
