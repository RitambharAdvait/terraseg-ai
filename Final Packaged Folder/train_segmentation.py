"""
Segmentation Training Script (IoU BOOST VERSION)
Goal: Achieve IoU 0.50+
Compatible with test_segmentation.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm


# ============================================================================
# Mask Conversion
# ============================================================================

value_map = {
    0: 0,
    100: 1,
    200: 2,
    300: 3,
    500: 4,
    550: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}

n_classes = len(value_map)


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.masks_dir = os.path.join(data_dir, "Segmentation")
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


# ============================================================================
# Model: Segmentation Head (128 channels SAME as test script)
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ============================================================================
# Dice Loss
# ============================================================================

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=n_classes).permute(0, 3, 1, 2).float()

    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float("nan"))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class)


def compute_pixel_accuracy(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model, backbone, loader, device):
    model.eval()
    iou_scores = []
    acc_scores = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device).squeeze(1).long()

            output = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(output)
            logits = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            iou_scores.append(compute_iou(logits, labels, num_classes=n_classes))
            acc_scores.append(compute_pixel_accuracy(logits, labels))

    model.train()
    return np.mean(iou_scores), np.mean(acc_scores)


# ============================================================================
# Main Training
# ============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # ============================
    # Hyperparameters (IoU Boost)
    # ============================
    batch_size = 6
    lr = 2e-4
    n_epochs = 80   # BEST FOR IoU 0.5+

    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ============================
    # Strong but SAFE Augmentation
    # ============================
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.05),
        transforms.RandomAutocontrast(p=0.25),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=Image.NEAREST),
        transforms.PILToTensor()
    ])

    train_dir = os.path.join(script_dir, "..", "Offroad_Segmentation_Training_Dataset", "train")
    val_dir = os.path.join(script_dir, "..", "Offroad_Segmentation_Training_Dataset", "val")

    trainset = MaskDataset(train_dir, transform=transform, mask_transform=mask_transform)
    valset = MaskDataset(val_dir, transform=transform, mask_transform=mask_transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")

    # ============================
    # Load Backbone
    # ============================
    print("\nLoading DINOv2 backbone...")
    backbone_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone_model.eval()
    backbone_model.to(device)
    print("Backbone loaded successfully!")

    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]

    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")

    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    ).to(device)

    # ============================
    # Weighted CrossEntropy (Boost IoU)
    # ============================
    class_weights = torch.tensor(
        [1.0, 2.0, 2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 1.5, 1.5],
        device=device
    )

    ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    scaler = torch.cuda.amp.GradScaler()

    best_val_iou = 0.0

    print("\nStarting IoU BOOST Training...")
    print("=" * 80)

    for epoch in range(n_epochs):
        classifier.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", unit="batch")

        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device).squeeze(1).long()

            with torch.no_grad():
                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]

            with torch.cuda.amp.autocast():
                logits = classifier(output)
                logits = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

                loss1 = ce_loss(logits, labels)
                loss2 = dice_loss(logits, labels)
                loss = loss1 + 1.0 * loss2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # ============================
        # Validation every epoch
        # ============================
        val_iou, val_acc = evaluate(classifier, backbone_model, val_loader, device)

        print("\n" + "-" * 80)
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train Loss: {np.mean(train_losses):.4f}")
        print(f"Val IoU:    {val_iou:.4f}")
        print(f"Val Acc:    {val_acc:.4f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 80)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(classifier.state_dict(), os.path.join(script_dir, "best_segmentation_head.pth"))
            print(f"🔥 BEST MODEL SAVED | Epoch {epoch+1} | Val IoU = {val_iou:.4f}")

    torch.save(classifier.state_dict(), os.path.join(script_dir, "segmentation_head.pth"))

    print("\nTraining complete!")
    print(f"Best Val IoU achieved: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
