"""
Single-image inference script for leaf disease detection.

Run disease prediction and optional Grad-CAM explainability on any leaf image:

    python predict.py --image path/to/leaf.jpg --checkpoint runs/baseline_full/best_model.pt
    python predict.py --image path/to/leaf.jpg --checkpoint runs/baseline_full/best_model.pt --cam
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.data.dataset import default_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.models.classifier import LeafClassifier
from src.interpretability.gradcam import GradCAM, get_last_conv_layer


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor back to [0, 1] RGB for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def cam_to_bbox(cam: torch.Tensor, threshold: float = 0.5) -> Optional[Tuple[int, int, int, int]]:
    """Extract largest connected component above threshold as (x, y, w, h) bounding box."""
    mask = (cam.cpu().numpy() > threshold).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h, _ = stats[largest_idx]
    return int(x), int(y), int(w), int(h)


def load_classes(checkpoint_path: Path, classes_file: Optional[Path] = None) -> List[str]:
    """Find class names from explicit file or next to checkpoint."""
    if classes_file and classes_file.exists():
        return json.loads(classes_file.read_text())
    
    adjacent = checkpoint_path.parent / "classes.json"
    if adjacent.exists():
        return json.loads(adjacent.read_text())
    
    return []


def predict_image(
    image_path: str,
    checkpoint_path: str,
    classes_file: Optional[str] = None,
    backbone: str = "efficientnet_b0",
    device: Optional[str] = None,
    top_k: int = 3,
    save_cam: bool = False,
    cam_out: Optional[str] = None,
) -> List[Tuple[str, float]]:
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Image not found: {img_p}")

    classes = load_classes(ckpt, Path(classes_file) if classes_file else None)
    
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model weights
    state_dict = torch.load(ckpt, map_location=dev)
    
    # Infer number of classes if classes.json was not found
    if not classes:
        # Check last linear layer weight shape in state_dict
        for key in reversed(list(state_dict.keys())):
            if "weight" in key and len(state_dict[key].shape) == 2:
                num_classes = state_dict[key].shape[0]
                classes = [f"Class_{i}" for i in range(num_classes)]
                break
    
    model = LeafClassifier(num_classes=len(classes), backbone=backbone, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(dev).eval()

    # Preprocess image
    raw_img = Image.open(img_p).convert("RGB")
    transform = default_transforms(image_size=224, train=False)
    img_tensor = transform(raw_img).unsqueeze(0).to(dev)

    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(classes)))
    results = [(classes[idx.item()], prob.item()) for idx, prob in zip(top_indices, top_probs)]

    # Optional Grad-CAM visualization
    if save_cam:
        target_layer = get_last_conv_layer(model)
        gradcam = GradCAM(model, target_layer)
        top_class_idx = top_indices[0].item()
        
        cam, _ = gradcam(img_tensor, class_idx=top_class_idx)
        denorm = denormalize(img_tensor.squeeze(0).cpu())
        bbox = cam_to_bbox(cam)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        img_np = denorm.permute(1, 2, 0).numpy()

        axes[0].imshow(img_np)
        axes[0].set_title(f"Input: {img_p.name}")
        axes[0].axis("off")

        axes[1].imshow(img_np)
        axes[1].imshow(cam.cpu().numpy(), cmap="jet", alpha=0.5)
        top_name, top_conf = results[0]
        axes[1].set_title(f"Grad-CAM: {top_name}\n({top_conf * 100:.1f}%)")
        axes[1].axis("off")

        axes[2].imshow(img_np)
        if bbox is not None:
            x, y, w, h = bbox
            rect = plt.Rectangle((x, y), w, h, edgecolor="lime", facecolor="none", linewidth=2)
            axes[2].add_patch(rect)
            axes[2].set_title(f"Pseudo Box: [{x}, {y}, {w}, {h}]")
        else:
            axes[2].set_title("No High-Confidence Region")
        axes[2].axis("off")

        out_path = Path(cam_out) if cam_out else img_p.parent / f"cam_{img_p.stem}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Grad-CAM visualization saved to: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Predict leaf disease on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image (JPG/PNG)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt checkpoint")
    parser.add_argument("--classes-file", type=str, default=None, help="Optional path to classes.json")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", help="Model backbone")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda or cpu)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to display")
    parser.add_argument("--cam", action="store_true", help="Generate Grad-CAM heatmap visualization")
    parser.add_argument("--cam-out", type=str, default=None, help="Path to save Grad-CAM visualization")
    args = parser.parse_args()

    results = predict_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        classes_file=args.classes_file,
        backbone=args.backbone,
        device=args.device,
        top_k=args.top_k,
        save_cam=args.cam,
        cam_out=args.cam_out,
    )

    print("\n--- Predictions ---")
    for rank, (class_name, confidence) in enumerate(results, 1):
        print(f"  {rank}. {class_name:35s}  {confidence * 100:6.2f}%")
    print("-------------------")


if __name__ == "__main__":
    main()

