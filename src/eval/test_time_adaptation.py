"""
Can cheap test-time methods close the PlantVillage -> PlantDoc gap of the 38-class model, without
retraining and without PlantDoc labels? Four methods, fixed in advance (nothing is tuned on PlantDoc test):

  baseline       the trained model as is
  tta            softmax averaged over 4 flips (none, horizontal, vertical, both), the flips used in training
  adabn          AdaBN (Li et al., 2016): BatchNorm statistics re-estimated on UNLABELED PlantDoc *train*
                 images; weights unchanged
  adabn_tta      both

AdaBN adapts on PlantDoc's train split only. PlantDoc train contains some byte-identical copies of test
images; those are excluded, so no test image is ever seen before it is scored. Each method is scored on:
  plantdoc_test      the 27 PlantDoc classes with a PlantVillage equivalent (same set as cross_dataset_eval --subset full)
  plantvillage_val   the same 27 classes of the PlantVillage val split, to show what adaptation costs in-distribution

Example:
    python -m src.eval.test_time_adaptation --checkpoint runs/baseline_full/best_model.pt \
        --plantdoc-root PlantDoc-Dataset --plantvillage-val-dir data/plantvillage_full/val \
        --output-dir runs/test_time_adaptation
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import LeafDataset, PathListDataset, default_transforms, set_seed
from src.eval.cross_dataset_eval import load_model
from src.eval.plantdoc import PlantDocDataset, PLANTDOC_FULL_TO_PLANTVILLAGE
from src.utils.metrics import classification_metrics, mcnemar

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
METHODS = ["baseline", "tta", "adabn", "adabn_tta"]


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def flips(images):
    """The 4 flip views of a batch (identity, horizontal, vertical, both)."""
    return [images, images.flip(-1), images.flip(-2), images.flip(-1).flip(-2)]


@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval()
    preds, labels = [], []
    for images, y in loader:
        images = images.to(device)
        views = flips(images) if tta else [images]
        probs = torch.stack([model(v).softmax(dim=1) for v in views]).mean(dim=0)
        preds.append(probs.argmax(dim=1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def adabn(model, loader, device):
    """Copy of `model` whose BatchNorm running mean/var are the plain average over `loader` (weights unchanged)."""
    adapted = copy.deepcopy(model)
    adapted.eval()  # dropout etc. stay in eval mode; only BatchNorm collects statistics
    bns = [m for m in adapted.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative moving average = exact mean over all batches
        bn.train()
    for images, _ in loader:
        adapted(images.to(device))
    adapted.eval()
    return adapted


def run(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.checkpoint, args.backbone, device)
    class_to_idx = {name: i for i, name in enumerate(classes)}
    tf = default_transforms(train=False, image_size=args.image_size)
    loader = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    pd_root = Path(args.plantdoc_root)
    test_ds = PlantDocDataset(pd_root / "test", class_to_idx, mapping=PLANTDOC_FULL_TO_PLANTVILLAGE, transform=tf)
    evaluated = sorted({classes[label] for _, label in test_ds.samples}, key=class_to_idx.get)
    pv_ds = LeafDataset(args.plantvillage_val_dir, transform=tf, classes=evaluated, class_to_idx=class_to_idx)

    # Unlabeled adaptation set: every PlantDoc train image except exact copies of test images.
    test_hashes = {md5(p) for p in (pd_root / "test").rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    train_files = sorted(p for p in (pd_root / "train").rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    adapt_files = [p for p in train_files if md5(p) not in test_hashes]
    adapt_ds = PathListDataset([(str(p), 0) for p in adapt_files], tf)  # label unused
    print(f"PlantDoc test: {len(test_ds)} images, {len(evaluated)} classes | PlantVillage val: {len(pv_ds)} | "
          f"AdaBN set: {len(adapt_files)} unlabeled PlantDoc train images "
          f"({len(train_files) - len(adapt_files)} duplicates of test images excluded)")

    adapted = adabn(model, loader(adapt_ds), device)
    results = {"config": {"classes_evaluated": len(evaluated), "n_plantdoc_test": len(test_ds),
                          "n_plantvillage_val": len(pv_ds), "n_adaptation_images": len(adapt_files),
                          "n_duplicates_excluded": len(train_files) - len(adapt_files),
                          "tta_views": "identity, hflip, vflip, hvflip"},
               "methods": {}}
    base_correct = None
    for method in METHODS:
        net = adapted if method.startswith("adabn") else model
        tta = method.endswith("tta")
        scores = {}
        for split, ds in (("plantdoc_test", test_ds), ("plantvillage_val", pv_ds)):
            preds, labels = predict(net, loader(ds), device, tta=tta)
            scores[split] = classification_metrics(labels, preds)
            if split == "plantdoc_test":
                correct = preds == labels
                if base_correct is None:
                    base_correct = correct
                else:
                    scores["plantdoc_test_vs_baseline"] = mcnemar(base_correct, correct)
        results["methods"][method] = scores
        vs = scores.get("plantdoc_test_vs_baseline")
        print(f"{method:10s}  PlantDoc test acc={scores['plantdoc_test']['accuracy']:.4f} "
              f"f1={scores['plantdoc_test']['macro_f1']:.4f}"
              + (f" (+{vs['gained']}/-{vs['lost']} images vs baseline, McNemar p={vs['p_value']:.3f})" if vs else "")
              + f"  |  PlantVillage val acc={scores['plantvillage_val']['accuracy']:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tta_results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"Wrote {out_dir / 'tta_results.json'}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="38-class model (baseline_full/best_model.pt)")
    p.add_argument("--plantdoc-root", required=True, help="PlantDoc clone, containing train/ and test/")
    p.add_argument("--plantvillage-val-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
