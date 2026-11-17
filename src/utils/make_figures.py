"""
Build the README sample figures from the saved checkpoints, so every image in assets/ comes from code:

  <run-dir>/figures/seg_samples.png            U-Net: input | ground truth | prediction (2 val samples)
  <run-dir>/figures/gradcam_comparison.png     Grad-CAM heatmap + pseudo box, one Apple disease per row
  <run-dir>/figures/gan_real_vs_generated.png  8 real vs. 8 DCGAN-generated Cedar apple rust leaves

Nothing is trained. Each figure is skipped (with a message) if its checkpoint or data is missing.
generate_report.py prefers these over the plain per-script figures.

    python -m src.utils.make_figures --run-dir ~/runs/plant_leaf --data-dir ~/data/plantvillage
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import random_split
import torchvision.transforms as T

from src.data.dataset import APPLE_CLASSES, default_transforms, set_seed
from src.interpretability.gradcam import GradCAM, get_last_conv_layer
from src.interpretability.pseudo_boxes import cam_to_bbox, denormalize
from src.models.classifier import LeafClassifier, load_checkpoint_classes
from src.models.gan import Generator
from src.models.unet import UNet
from src.training.train_segmentation import SegmentationDataset
from src.utils.figure_style import INK, BLUE_600, GREEN_BOX

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# One row per disease (healthy leaves have no lesion to localize).
GRADCAM_ROWS = [("Apple___Apple_scab", "Apple Scab"), ("Apple___Black_rot", "Apple Black Rot"),
                ("Apple___Cedar_apple_rust", "Apple Cedar Rust")]


def _images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def seg_figure(args, device, out_path):
    ckpt = args.run_dir / "unet" / "unet_best.pt"
    results = json.loads((args.run_dir / "unet" / "seg_results.json").read_text())
    dataset = SegmentationDataset(args.data_dir / "seg" / "images", args.data_dir / "seg" / "masks",
                                  args.seg_image_size)
    # Same split as train_segmentation.train(), so these are validation images.
    val_len = max(1, int(0.1 * len(dataset)))
    _, val_ds = random_split(dataset, [len(dataset) - val_len, val_len],
                             generator=torch.Generator().manual_seed(42))
    model = UNet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    n = min(2, len(val_ds))
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.45 * n + 0.6), squeeze=False, facecolor="white")
    with torch.no_grad():
        for i in range(n):
            image, mask = val_ds[i]
            pred = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0].cpu() > 0.5
            img = (image * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
            for j, data in enumerate([img, mask[0].numpy(), pred.numpy()]):
                ax = axes[i][j]
                ax.imshow(data, cmap=None if data.ndim == 3 else "gray", vmin=0, vmax=1)
                ax.axis("off")
                if i == 0:
                    ax.set_title(["Input Leaf", "Ground Truth Mask", "U-Net Prediction"][j],
                                 fontsize=12, fontweight="bold", color=INK, pad=8)
    fig.suptitle(f"U-Net Leaf Segmentation Samples (Val IoU: {results['val_iou']:.3f} · "
                 f"Val Dice: {results['val_dice']:.3f})", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gradcam_figure(args, device, out_path):
    ckpt = args.run_dir / "baseline_apple_only" / "best_model.pt"
    load_checkpoint_classes(ckpt, APPLE_CLASSES)
    model = LeafClassifier(num_classes=len(APPLE_CLASSES), backbone="efficientnet_b0", pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    cam_extractor = GradCAM(model, get_last_conv_layer(model))
    transform = default_transforms(train=False, image_size=args.image_size)

    fig, axes = plt.subplots(len(GRADCAM_ROWS), 3, figsize=(10, 3.2 * len(GRADCAM_ROWS) + 0.7),
                             squeeze=False, facecolor="white")
    for i, (folder, label) in enumerate(GRADCAM_ROWS):
        # First validation image of the class (sorted), so the figure is the same on every run.
        path = _images(args.data_dir / "apple" / "val" / folder)[0]
        tensor = transform(Image.open(path).convert("RGB"))
        cam, pred = cam_extractor(tensor.unsqueeze(0).to(device))  # explains the predicted class
        cam = cam.cpu()
        img = denormalize(tensor).permute(1, 2, 0).numpy()
        bbox = cam_to_bbox(cam, threshold=args.cam_threshold)

        axes[i][0].imshow(img)
        axes[i][1].imshow(img)
        axes[i][1].imshow(cam.numpy(), cmap="jet", alpha=0.5)
        axes[i][2].imshow(img)
        if bbox is not None:
            x, y, w, h = bbox
            axes[i][2].add_patch(plt.Rectangle((x, y), w, h, edgecolor=GREEN_BOX, facecolor="none", linewidth=2.5))
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        pred_name = APPLE_CLASSES[pred].replace("Apple___", "").replace("_", " ")
        axes[i][0].set_ylabel(f"{label}\n(predicted: {pred_name})", rotation=0, ha="right", va="center",
                              fontsize=11, fontweight="bold", color=INK, labelpad=12)
    for j, title in enumerate(["Input Leaf", "Grad-CAM Heatmap", "Lesion Bounding Box"]):
        axes[0][j].set_title(title, fontsize=12, fontweight="bold", color=INK, pad=10)
    fig.suptitle("Weakly-Supervised Lesion Localization (Grad-CAM to Pseudo Bounding Box)",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gan_figure(args, device, out_path):
    ckpt = args.run_dir / "gan_cedar_rust" / "generator.pt"
    real_dir = args.data_dir / "apple" / "train" / "Apple___Cedar_apple_rust"
    netG = Generator(latent_dim=args.latent_dim).to(device)
    netG.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    netG.eval()

    size = args.gan_image_size
    to_tensor = T.Compose([T.Resize((size, size)), T.ToTensor()])
    real = torch.stack([to_tensor(Image.open(p).convert("RGB")) for p in _images(real_dir)[:8]])
    noise = torch.randn(8, args.latent_dim, 1, 1, generator=torch.Generator().manual_seed(0)).to(device)
    with torch.no_grad():
        fake = ((netG(noise).cpu() + 1) / 2).clamp(0, 1)  # generator outputs tanh in [-1, 1]

    def strip(batch):
        pad = 2
        row = torch.ones(3, size + 2 * pad, len(batch) * (size + pad) + pad) * 0.0
        for k, im in enumerate(batch):
            x = pad + k * (size + pad)
            row[:, pad:pad + size, x:x + size] = im
        return row.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.3), facecolor="white")
    for ax, batch, title, color in [(axes[0], real, "Real Training Leaves (Apple Cedar Rust)", INK),
                                    (axes[1], fake, f"DCGAN Synthetic Generated Leaves ({size}×{size})", BLUE_600)]:
        ax.imshow(strip(batch))
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", color=color, pad=6)
    fig.suptitle("DCGAN Minority Class Synthesis: Real vs. Generated", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir or args.run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in [("seg_samples.png", seg_figure), ("gradcam_comparison.png", gradcam_figure),
                     ("gan_real_vs_generated.png", gan_figure)]:
        try:
            fn(args, device, out_dir / name)
            print(f"wrote {out_dir / name}")
        except (FileNotFoundError, IndexError) as e:  # step not run yet / no data for it
            print(f"skipped {name}: {e}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path, help="prepared data (apple/, seg/)")
    p.add_argument("--output-dir", type=Path, default=None, help="default: <run-dir>/figures")
    p.add_argument("--image-size", type=int, default=224, help="classifier input size (Grad-CAM)")
    p.add_argument("--seg-image-size", type=int, default=256, help="U-Net input size")
    p.add_argument("--gan-image-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--cam-threshold", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
