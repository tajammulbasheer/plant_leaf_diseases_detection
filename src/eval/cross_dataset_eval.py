"""
Evaluate a trained classifier on both:
  (a) the PlantVillage validation set it was trained/validated on (in-distribution)
  (b) PlantDoc, a real-world field-photo dataset (out-of-distribution)

...and report the accuracy/F1 gap between them. This is the headline result
for the "cross-dataset generalization" experiment: a large gap tells you the
model learned lab-condition shortcuts (clean backgrounds, consistent framing)
rather than the actual disease texture.

Two subsets:
  --subset apple (default)  the 4-class Apple model on PlantDoc's 3 Apple folders
  --subset full             the 38-class model on every PlantDoc class that has a
                            PlantVillage equivalent; the in-distribution score uses
                            the same classes of the PlantVillage val split, and a
                            per-class chart replaces the (38x38) confusion matrices

Example (Colab):
    !python -m src.eval.cross_dataset_eval \
        --checkpoint /content/drive/MyDrive/apple_leaf_runs/baseline_apple_only/best_model.pt \
        --plantvillage-val-dir /content/data/apple/val \
        --plantdoc-dir /content/PlantDoc-Dataset/test \
        --output-dir /content/drive/MyDrive/apple_leaf_runs/cross_dataset_eval

    !python -m src.eval.cross_dataset_eval --subset full \
        --checkpoint /content/drive/MyDrive/apple_leaf_runs/baseline_full/best_model.pt \
        --plantvillage-val-dir /content/data/plantvillage_full/val \
        --plantdoc-dir /content/PlantDoc-Dataset/test \
        --output-dir /content/drive/MyDrive/apple_leaf_runs/cross_dataset_eval_full
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.data.dataset import AppleLeafDataset, LeafDataset, APPLE_CLASSES, default_transforms
from src.eval.plantdoc import PlantDocAppleDataset, PlantDocDataset, PLANTDOC_FULL_TO_PLANTVILLAGE
from src.models.classifier import LeafClassifier, load_checkpoint_classes
from src.utils.metrics import evaluate_classifier
from src.utils.figure_style import BLUE_RAMP, BLUE_600, ORANGE_600, INK, INK_2, MUTED, BORDER


def load_model(checkpoint_path, backbone, device, expected_classes=None):
    """Returns (model, classes). Apple mode checks the checkpoint really has the 4 Apple classes."""
    classes = load_checkpoint_classes(checkpoint_path, expected_classes) or expected_classes
    if not classes:
        raise FileNotFoundError(f"No classes.json next to {checkpoint_path}; cannot map outputs to classes.")
    model = LeafClassifier(num_classes=len(classes), backbone=backbone, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model, list(classes)


@torch.no_grad()
def predict(model, loader, device):
    preds, labels = [], []
    for images, y in loader:
        preds.append(model(images.to(device)).argmax(dim=1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


APPLE_SHORT = {"Apple___Apple_scab": "Apple Scab", "Apple___Black_rot": "Black Rot",
               "Apple___Cedar_apple_rust": "Cedar Rust", "Apple___healthy": "Healthy"}


def plot_confusion(model, loader, device, class_names, title, save_path, subtitle="", acc_label="Accuracy"):
    all_preds, all_labels = predict(model, loader, device)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    clean_classes = [APPLE_SHORT.get(c, c.replace("_", " ")) for c in class_names]
    # Counts come from the matrix itself, so the subtitle can never disagree with it.
    correct, total = int(np.trace(cm)), int(cm.sum())
    if subtitle:
        subtitle = f"{subtitle} · {acc_label}: {correct / max(total, 1) * 100:.1f}% ({correct}/{total})"

    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="white")
    ax.set_facecolor("white")

    ax.imshow(cm, interpolation="nearest", cmap=BLUE_RAMP)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98, color=INK)
    if subtitle:
        ax.set_title(subtitle, fontsize=10, color=MUTED, pad=10)

    ax.set_xticks(range(len(clean_classes)))
    ax.set_yticks(range(len(clean_classes)))
    ax.set_xticklabels(clean_classes, fontsize=9.5, color=INK_2, rotation=25, ha="right")
    ax.set_yticklabels(clean_classes, fontsize=9.5, color=INK_2)

    ax.set_xlabel("Predicted Class", fontsize=10.5, fontweight="bold", color=INK, labelpad=8)
    ax.set_ylabel("True Class", fontsize=10.5, fontweight="bold", color=INK, labelpad=8)

    ax.set_xticks(np.arange(-0.5, len(clean_classes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(clean_classes), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0)

    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(1.2)

    thresh = cm.max() * 0.45 if cm.max() > 0 else 1
    for i in range(len(clean_classes)):
        for j in range(len(clean_classes)):
            val = cm[i, j]
            color = "#ffffff" if val > thresh else INK
            weight = "bold" if val > 0 else "normal"
            ax.text(j, i, f"{val}", ha="center", va="center",
                    fontsize=11.5, fontweight=weight, color=color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def per_class_accuracy(preds, labels, idx_to_name):
    out = {}
    for idx in sorted(set(labels.tolist())):
        mask = labels == idx
        out[idx_to_name[idx]] = {"accuracy": float((preds[mask] == idx).mean()), "n": int(mask.sum())}
    return out


def plot_per_class(per_class, save_path):
    """Horizontal bars: per-class accuracy on PlantVillage val vs. PlantDoc, PlantDoc image count in the label."""
    names = sorted(per_class, key=lambda c: per_class[c]["plantdoc"])
    labels = [f"{c.replace('___', ': ').replace('_', ' ')}  (n={per_class[c]['n_plantdoc']})" for c in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(names) + 1.4), facecolor="white")
    ax.barh(y + 0.2, [per_class[c]["plantvillage"] for c in names], 0.4, color=BLUE_600,
            label="PlantVillage val (lab photos)", zorder=3)
    ax.barh(y - 0.2, [per_class[c]["plantdoc"] for c in names], 0.4, color=ORANGE_600,
            label="PlantDoc test (field photos)", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK_2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Per-class accuracy (recall)", fontsize=9.5, fontweight="bold", color=INK)
    ax.xaxis.grid(True, color="#e2e8f0", linestyle="--", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BORDER)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("38-class model: per-class accuracy, lab vs. field photos", fontsize=11,
                 fontweight="bold", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_apple(args, device, out_dir):
    class_to_idx = {name: i for i, name in enumerate(APPLE_CLASSES)}
    model, _ = load_model(args.checkpoint, args.backbone, device, APPLE_CLASSES)
    tf = default_transforms(train=False, image_size=args.image_size)

    # --- In-distribution: PlantVillage val ---
    pv_ds = AppleLeafDataset(root=args.plantvillage_val_dir, transform=tf)
    pv_loader = DataLoader(pv_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    pv_metrics = evaluate_classifier(model, pv_loader, device, APPLE_CLASSES)
    plot_confusion(model, pv_loader, device, APPLE_CLASSES, "PlantVillage (In-Distribution)",
                   out_dir / "confusion_plantvillage.png", subtitle="Apple Subset", acc_label="Val Accuracy")

    # --- Out-of-distribution: PlantDoc ---
    pd_ds = PlantDocAppleDataset(root=args.plantdoc_dir, class_to_idx=class_to_idx, transform=tf)
    print(pd_ds.missing_classes_note)
    pd_loader = DataLoader(pd_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    pd_metrics = evaluate_classifier(model, pd_loader, device, APPLE_CLASSES)
    plot_confusion(model, pd_loader, device, APPLE_CLASSES, "PlantDoc (Field Photos · OOD)",
                   out_dir / "confusion_plantdoc.png", subtitle="Apple Subset")

    return {
        "plantvillage_in_distribution": pv_metrics,
        "plantdoc_out_of_distribution": pd_metrics,
        "accuracy_gap": pv_metrics["accuracy"] - pd_metrics["accuracy"],
        "macro_f1_gap": pv_metrics["macro_f1"] - pd_metrics["macro_f1"],
        "note": pd_ds.missing_classes_note,
    }


def run_full(args, device, out_dir):
    model, classes = load_model(args.checkpoint, args.backbone, device)
    class_to_idx = {name: i for i, name in enumerate(classes)}
    idx_to_name = {i: name for name, i in class_to_idx.items()}
    tf = default_transforms(train=False, image_size=args.image_size)

    # --- Out-of-distribution: every PlantDoc folder that maps to one of the model's classes ---
    pd_ds = PlantDocDataset(root=args.plantdoc_dir, class_to_idx=class_to_idx,
                            mapping=PLANTDOC_FULL_TO_PLANTVILLAGE, transform=tf)
    print(pd_ds.missing_classes_note)
    pd_loader = DataLoader(pd_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    pd_preds, pd_labels = predict(model, pd_loader, device)
    evaluated = sorted({idx_to_name[i] for i in pd_labels.tolist()}, key=class_to_idx.get)

    # --- In-distribution: the same classes of the PlantVillage val split (the model still predicts all classes) ---
    pv_ds = LeafDataset(args.plantvillage_val_dir, transform=tf, classes=evaluated, class_to_idx=class_to_idx)
    pv_loader = DataLoader(pv_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    pv_preds, pv_labels = predict(model, pv_loader, device)

    pv_metrics = evaluate_classifier(model, pv_loader, device, classes)
    pd_metrics = evaluate_classifier(model, pd_loader, device, classes)
    pv_metrics["n_images"], pd_metrics["n_images"] = len(pv_ds), len(pd_ds)

    pv_pc = per_class_accuracy(pv_preds, pv_labels, idx_to_name)
    pd_pc = per_class_accuracy(pd_preds, pd_labels, idx_to_name)
    per_class = {c: {"plantvillage": pv_pc[c]["accuracy"], "n_plantvillage": pv_pc[c]["n"],
                     "plantdoc": pd_pc[c]["accuracy"], "n_plantdoc": pd_pc[c]["n"]}
                 for c in evaluated if c in pv_pc}
    plot_per_class(per_class, out_dir / "per_class_accuracy.png")

    return {
        "subset": "full",
        "classes_evaluated": len(evaluated),
        "plantvillage_in_distribution": pv_metrics,
        "plantdoc_out_of_distribution": pd_metrics,
        "accuracy_gap": pv_metrics["accuracy"] - pd_metrics["accuracy"],
        "macro_f1_gap": pv_metrics["macro_f1"] - pd_metrics["macro_f1"],
        "per_class": per_class,
        "note": (f"{len(evaluated)} PlantDoc classes with a PlantVillage equivalent, {len(pd_ds)} PlantDoc "
                 f"test images; the model predicts over all {len(classes)} classes."),
    }


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_full(args, device, out_dir) if args.subset == "full" else run_apple(args, device, out_dir)
    with open(out_dir / "cross_dataset_results.json", "w") as f:
        json.dump(results, f, indent=2)

    pv_metrics, pd_metrics = results["plantvillage_in_distribution"], results["plantdoc_out_of_distribution"]
    print(f"\nIn-distribution (PlantVillage): acc={pv_metrics['accuracy']:.4f} f1={pv_metrics['macro_f1']:.4f}")
    print(f"Out-of-distribution (PlantDoc): acc={pd_metrics['accuracy']:.4f} f1={pd_metrics['macro_f1']:.4f}")
    print(f"Generalization gap: {results['accuracy_gap']:.4f} accuracy, {results['macro_f1_gap']:.4f} macro-F1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--plantvillage-val-dir", required=True)
    p.add_argument("--plantdoc-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subset", choices=["apple", "full"], default="apple",
                   help="apple: 4-class Apple model (default); full: 38-class model on all overlapping classes")
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
