"""
Segmentation Validation Script
Converted from val_mask.ipynb
Evaluates a trained segmentation head on validation data and saves predictions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

n_classes = len(value_map)

color_palette = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)


def convert_mask(mask):
    """Convert raw mask values (100,200,...) into class IDs (0-9)."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)

    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value

    return Image.fromarray(new_arr)


def mask_to_color(mask):
    """Convert a class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]

    return color_mask


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
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

        # IMPORTANT: convert raw mask values into class IDs 0-9
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        # mask shape = [1,H,W]
        return image, mask, data_id


# ============================================================================
# Model: Segmentation Head (ConvNeXt-style) - MUST match training code
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

    return np.nanmean(iou_per_class), iou_per_class


def compute_pixel_accuracy(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


# ============================================================================
# Visualization Functions
# ============================================================================

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    """Save side-by-side input image + ground truth + prediction."""
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(gt_color)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_color)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.suptitle(f"Sample: {data_id}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_metrics_summary(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, "evaluation_metrics.txt")

    with open(filepath, "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean IoU: {results['mean_iou']:.4f}\n")
        f.write(f"Mean Accuracy: {results['mean_acc']:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Class IoU:\n")
        f.write("-" * 40 + "\n")
        for name, iou in zip(class_names, results["class_iou"]):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"{name:<20}: {iou_str}\n")

    print(f"\nSaved evaluation metrics to {filepath}")

    # Save bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results["class_iou"]]
    ax.bar(range(n_classes), valid_iou)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("IoU")
    ax.set_title(f"Per-Class IoU (Mean: {results['mean_iou']:.4f})")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved per-class metrics chart to '{output_dir}/per_class_metrics.png'")


# ============================================================================
# Main Validation Function
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Segmentation prediction/inference script")

    parser.add_argument("--model_path", type=str,
                        default=os.path.join(script_dir, "segmentation_head.pth"),
                        help="Path to trained model weights")

    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(script_dir, "..", "Offroad_Segmentation_testImages"),
                        help="Path to test dataset")

    parser.add_argument("--output_dir", type=str,
                        default="./predictions",
                        help="Directory to save prediction visualizations")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=5)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # IMPORTANT: PILToTensor keeps mask values EXACT (0-9)
    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=Image.NEAREST),
        transforms.PILToTensor()
    ])

    print(f"Loading dataset from {args.data_dir}...")
    valset = MaskDataset(data_dir=args.data_dir, transform=transform, mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(valset)} samples")

    # ==========================================================
    # Load DINOv2 backbone
    # ==========================================================
    print("Loading DINOv2 backbone...")

    BACKBONE_SIZE = "small"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }

    backbone_name = f"dinov2_{backbone_archs[BACKBONE_SIZE]}"

    backbone_model = torch.hub.load("facebookresearch/dinov2", backbone_name)
    backbone_model.eval()
    backbone_model.to(device)

    print("Backbone loaded successfully!")

    # Get embedding dim
    sample_img, _, _ = valset[0]
    sample_img = sample_img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = backbone_model.forward_features(sample_img)["x_norm_patchtokens"]

    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")

    # ==========================================================
    # Load classifier
    # ==========================================================
    print(f"Loading model from {args.model_path}...")

    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    ).to(device)

    classifier.load_state_dict(torch.load(args.model_path, map_location=device))
    classifier.eval()

    print("Model loaded successfully!")

    # Create output subfolders
    masks_dir = os.path.join(args.output_dir, "masks")
    masks_color_dir = os.path.join(args.output_dir, "masks_color")
    comparisons_dir = os.path.join(args.output_dir, "comparisons")

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    print(f"\nRunning evaluation and saving predictions for all {len(valset)} images...")

    iou_scores = []
    acc_scores = []
    all_class_iou = []
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Processing", unit="batch")

        for imgs, labels, data_ids in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(output)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels_squeezed = labels.squeeze(dim=1).long()
            predicted_masks = torch.argmax(outputs, dim=1)

            mean_iou, class_iou = compute_iou(outputs, labels_squeezed, num_classes=n_classes)
            mean_acc = compute_pixel_accuracy(outputs, labels_squeezed)

            iou_scores.append(mean_iou)
            acc_scores.append(mean_acc)
            all_class_iou.append(class_iou)

            # Save predictions
            for i in range(imgs.shape[0]):
                data_id = data_ids[i]
                base_name = os.path.splitext(data_id)[0]

                pred_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)

                # raw mask
                Image.fromarray(pred_mask).save(os.path.join(masks_dir, f"{base_name}_pred.png"))

                # colored mask
                pred_color = mask_to_color(pred_mask)
                cv2.imwrite(
                    os.path.join(masks_color_dir, f"{base_name}_pred_color.png"),
                    cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
                )

                # save sample comparisons
                if sample_count < args.num_samples:
                    save_prediction_comparison(
                        imgs[i], labels_squeezed[i], predicted_masks[i],
                        os.path.join(comparisons_dir, f"sample_{sample_count}_comparison.png"),
                        data_id
                    )
                    sample_count += 1

            pbar.set_postfix(acc=f"{mean_acc:.3f}", iou=f"{mean_iou:.3f}")

    mean_iou = np.nanmean(iou_scores)
    mean_acc = np.mean(acc_scores)

    avg_class_iou = np.nanmean(all_class_iou, axis=0)

    results = {
        "mean_iou": mean_iou,
        "mean_acc": mean_acc,
        "class_iou": avg_class_iou
    }

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean IoU:          {mean_iou:.4f}")
    print(f"Mean Accuracy:     {mean_acc:.4f}")
    print("=" * 50)

    save_metrics_summary(results, args.output_dir)

    print(f"\nPrediction complete! Processed {len(valset)} images.")
    print(f"\nOutputs saved to {args.output_dir}/")
    print("  - masks/           : Raw prediction masks (class IDs 0-9)")
    print("  - masks_color/     : Colored prediction masks (RGB)")
    print(f"  - comparisons/     : Side-by-side comparison images ({args.num_samples} samples)")
    print("  - evaluation_metrics.txt")
    print("  - per_class_metrics.png")


if __name__ == "__main__":
    main()
