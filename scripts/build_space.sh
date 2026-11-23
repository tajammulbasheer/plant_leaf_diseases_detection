#!/usr/bin/env bash
# Assemble a Hugging Face Space (Gradio) for app.py: the code it imports, the sample images and the three
# trained checkpoints. Weights go to the Space only, never to this git repo.
#
#   scripts/build_space.sh                     # builds ~/projects/plant_leaf_space
#   SPACE_DIR=/tmp/space scripts/build_space.sh
#   hf upload <user>/<space-name> ~/projects/plant_leaf_space . --repo-type space
#
# Env vars: RUN_DIR (default ~/runs/plant_leaf), SPACE_DIR (default ~/projects/plant_leaf_space),
#           GITHUB_URL (linked from the Space page).
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${RUN_DIR:-$HOME/runs/plant_leaf}"
OUT="${SPACE_DIR:-$HOME/projects/plant_leaf_space}"
GITHUB_URL="${GITHUB_URL:-https://github.com/tajammulbasheer/plant_leaf_diseases_detection}"
GRADIO_VERSION="$(cd "$REPO" && "${PYTHON:-.venv/bin/python}" -c 'import gradio; print(gradio.__version__)')"

for f in baseline_apple_only/best_model.pt baseline_apple_only/classes.json \
         baseline_full/best_model.pt baseline_full/classes.json \
         baseline_full_bgrand/best_model.pt baseline_full_bgrand/classes.json; do
    [ -f "$RUN/$f" ] || { echo "missing $RUN/$f -- train the classifiers first (scripts/run_all.sh)"; exit 1; }
done

rm -rf "$OUT"
mkdir -p "$OUT"
cd "$REPO"
# Only the modules app.py imports (directly or through pseudo_boxes / dataset).
for f in app.py src/__init__.py \
         src/data/__init__.py src/data/dataset.py \
         src/models/__init__.py src/models/classifier.py \
         src/interpretability/__init__.py src/interpretability/gradcam.py src/interpretability/pseudo_boxes.py; do
    install -D -m 644 "$f" "$OUT/$f"
done
cp -r samples "$OUT/samples"
for m in baseline_apple_only baseline_full baseline_full_bgrand; do
    install -D -m 644 "$RUN/$m/best_model.pt" "$OUT/runs/$m/best_model.pt"
    install -D -m 644 "$RUN/$m/classes.json" "$OUT/runs/$m/classes.json"
done

# Default (CUDA) torch wheels: they also run on CPU Spaces, and ZeroGPU Spaces need them.
cat > "$OUT/requirements.txt" <<EOF
torch
torchvision
numpy
Pillow
matplotlib
opencv-python-headless
EOF

cat > "$OUT/.gitattributes" <<'EOF'
*.pt filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
EOF

cat > "$OUT/README.md" <<EOF
---
title: Leaf Disease Detection
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: $GRADIO_VERSION
app_file: app.py
pinned: false
license: mit
short_description: Leaf disease diagnosis with Grad-CAM lesion localization
---

# Leaf Disease Detection

Upload a leaf photo to get a diagnosis from an EfficientNet-B0 trained on PlantVillage (an Apple
specialist with 4 classes, the full 38-class model, or the 38-class model retrained with randomized
backgrounds), a Grad-CAM heatmap, and a pseudo bounding box
derived from it without any box labels.

**Limitation:** the models were trained on lab photos of single leaves on plain backgrounds. On real field
photos they are unreliable; the repository measures this on PlantDoc. The background-randomized model does
better there but is still far from field-ready.

Code, training pipeline and results: $GITHUB_URL
EOF

echo "Space assembled in $OUT ($(du -sh "$OUT" | cut -f1)), gradio $GRADIO_VERSION:"
(cd "$OUT" && find . -type f | sort | sed 's/^/  /')
