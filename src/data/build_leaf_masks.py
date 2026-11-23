"""
Leaf masks for every image of the prepared 38-class split, for background randomization
(train_classifier --background-dir / --mask-dir).

PlantVillage ships each photo a second time with the background blacked out (raw/segmented/). This matches
each image in <data-dir>/{train,val}/<class>/ to its segmented copy by file name and writes a binary mask
(255 = leaf) with the same mask_from_segmented() used for the U-Net data:

    <output-dir>/{train,val}/<class>/<image stem>.png

The split itself is not touched. Images with no segmented copy get no mask (they are then used unchanged).

    python -m src.data.build_leaf_masks --repo-dir ~/data/plantvillage/PlantVillage-Dataset \
        --data-dir ~/data/plantvillage/plantvillage_full --output-dir ~/data/plantvillage/plantvillage_full_masks
"""

import argparse
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

from src.data.prepare_plantvillage import _norm_stem, clone_repo, mask_from_segmented

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def run(args):
    repo_dir = Path(args.repo_dir)
    clone_repo(repo_dir)  # no-op if already cloned
    seg_root = repo_dir / "raw" / "segmented"
    if not seg_root.exists():
        raise FileNotFoundError(f"{seg_root} not found in the PlantVillage clone.")
    data_dir, out_dir = Path(args.data_dir), Path(args.output_dir)

    written = missing = empty = 0
    for split in ("train", "val"):
        for class_dir in sorted(d for d in (data_dir / split).iterdir() if d.is_dir()):
            seg_dir = seg_root / class_dir.name
            lookup = {_norm_stem(f.name): f for f in seg_dir.iterdir()} if seg_dir.exists() else {}
            dst = out_dir / split / class_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for img in tqdm(sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
                            desc=f"{split}/{class_dir.name}", leave=False):
                seg = lookup.get(_norm_stem(img.name))
                if seg is None:
                    missing += 1
                    continue
                mask = mask_from_segmented(seg)
                if mask is None or mask.mean() < 1:
                    empty += 1
                    continue
                cv2.imwrite(str(dst / f"{img.stem}.png"), mask)
                written += 1
    total = written + missing + empty
    print(f"Wrote {written} masks to {out_dir} ({written / max(total, 1) * 100:.1f}% of {total} images); "
          f"{missing} without a segmented copy, {empty} with an empty mask (used unchanged).")
    if args.cleanup_clone:
        print(f"Removing cloned repo at {repo_dir} to free disk space...")
        shutil.rmtree(repo_dir)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-dir", required=True, help="PlantVillage-Dataset clone (cloned here if missing)")
    p.add_argument("--cleanup-clone", action="store_true", help="Delete the clone afterwards")
    p.add_argument("--data-dir", required=True, help="prepared split, e.g. <data>/plantvillage_full")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
