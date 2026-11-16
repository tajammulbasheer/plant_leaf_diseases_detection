"""
Evaluate a trained classifier on both:
  (a) the PlantVillage validation set it was trained/validated on (in-distribution)
  (b) PlantDoc, a real-world field-photo dataset (out-of-distribution)

...and report the accuracy/F1 gap between them. This is the headline result
for the "cross-dataset generalization" experiment: a large gap tells you the
model learned lab-condition shortcuts (clean backgrounds, consistent framing)
rather than the actual disease texture.

Example (Colab):
    !python -m src.eval.cross_dataset_eval \
        --checkpoint /content/drive/MyDrive/apple_leaf_runs/baseline_full/best_model.pt \
        --plantvillage-val-dir /content/data/apple/val \
        --plantdoc-dir /content/PlantDoc-Dataset/test \
        --output-dir /content/drive/MyDrive/apple_leaf_runs/cross_dataset_eval
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.data.dataset import AppleLeafDataset, APPLE_CLASSES, default_transforms
from src.eval.plantdoc import PlantDocAppleDataset
from src.models.classifier import LeafClassifier, load_checkpoint_classes
from src.utils.metrics import evaluate_classifier


def load_model(checkpoint_path, backbone, device):
    load_checkpoint_classes(checkpoint_path, APPLE_CLASSES)
    model = LeafClassifier(num_classes=len(APPLE_CLASSES), backbone=backbone, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model


def plot_confusion(model, loader, device, class_names, title, save_path, subtitle=""):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    clean_classes = [c.replace("Apple___", "").replace("_", " ") for c in class_names]

    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    BLUE_RAMP = LinearSegmentedColormap.from_list(
        "blue_ramp", ["#f8fafc", "#bfdbfe", "#3b82f6", "#1d4ed8", "#172554"]
    )

    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="white")
    ax.set_facecolor("white")

    im = ax.imshow(cm, interpolation="nearest", cmap=BLUE_RAMP)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98, color="#0f172a")
    if subtitle:
        ax.set_title(subtitle, fontsize=10, color="#64748b", pad=10)

    ax.set_xticks(range(len(clean_classes)))
    ax.set_yticks(range(len(clean_classes)))
    ax.set_xticklabels(clean_classes, fontsize=9.5, color="#1e293b", rotation=25, ha="right")
    ax.set_yticklabels(clean_classes, fontsize=9.5, color="#1e293b")

    ax.set_xlabel("Predicted Class", fontsize=10.5, fontweight="bold", color="#0f172a", labelpad=8)
    ax.set_ylabel("True Class", fontsize=10.5, fontweight="bold", color="#0f172a", labelpad=8)

    ax.set_xticks(np.arange(-0.5, len(clean_classes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(clean_classes), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0)

    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(1.2)

    thresh = cm.max() * 0.45 if cm.max() > 0 else 1
    for i in range(len(clean_classes)):
        for j in range(len(clean_classes)):
            val = cm[i, j]
            color = "#ffffff" if val > thresh else "#0f172a"
            weight = "bold" if val > 0 else "normal"
            ax.text(j, i, f"{val}", ha="center", va="center",
                    fontsize=11.5, fontweight=weight, color=color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_to_idx = {name: i for i, name in enumerate(APPLE_CLASSES)}
    model = load_model(args.checkpoint, args.backbone, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- In-distribution: PlantVillage val ---
    pv_ds = AppleLeafDataset(
        root=args.plantvillage_val_dir,
        transform=default_transforms(train=False, image_size=args.image_size),
    )
    pv_loader = DataLoader(pv_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    pv_metrics = evaluate_classifier(model, pv_loader, device, APPLE_CLASSES)
    plot_confusion(
        model, pv_loader, device, APPLE_CLASSES,
        "PlantVillage (In-Distribution)", out_dir / "confusion_plantvillage.png",
        subtitle=f"Apple Subset · Val Accuracy: {pv_metrics['accuracy']*100:.1f}% ({len(pv_ds)}/{len(pv_ds)})",
    )

    # --- Out-of-distribution: PlantDoc ---
    pd_ds = PlantDocAppleDataset(
        root=args.plantdoc_dir,
        class_to_idx=class_to_idx,
        transform=default_transforms(train=False, image_size=args.image_size),
    )
    print(pd_ds.missing_classes_note)
    pd_loader = DataLoader(pd_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    pd_metrics = evaluate_classifier(model, pd_loader, device, APPLE_CLASSES)
    plot_confusion(
        model, pd_loader, device, APPLE_CLASSES,
        "PlantDoc (Field Photos · OOD)", out_dir / "confusion_plantdoc.png",
        subtitle=f"Apple Subset · Accuracy: {pd_metrics['accuracy']*100:.1f}% ({int(round(pd_metrics['accuracy']*len(pd_ds)))}/{len(pd_ds)})",
    )

    results = {
        "plantvillage_in_distribution": pv_metrics,
        "plantdoc_out_of_distribution": pd_metrics,
        "accuracy_gap": pv_metrics["accuracy"] - pd_metrics["accuracy"],
        "macro_f1_gap": pv_metrics["macro_f1"] - pd_metrics["macro_f1"],
        "note": pd_ds.missing_classes_note,
    }
    with open(out_dir / "cross_dataset_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nIn-distribution (PlantVillage): acc={pv_metrics['accuracy']:.4f} f1={pv_metrics['macro_f1']:.4f}")
    print(f"Out-of-distribution (PlantDoc): acc={pd_metrics['accuracy']:.4f} f1={pd_metrics['macro_f1']:.4f}")
    print(f"Generalization gap: {results['accuracy_gap']:.4f} accuracy, {results['macro_f1_gap']:.4f} macro-F1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--plantvillage-val-dir", required=True)
    p.add_argument("--plantdoc-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
