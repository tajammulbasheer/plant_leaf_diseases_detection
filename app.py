"""
Interactive Gradio Web Application for Leaf Disease Detection & Localization.

Features:
- Instant disease diagnosis with class confidence breakdown
- Weakly-supervised lesion localization via Grad-CAM heatmaps
- Automatic pseudo-bounding box detection around lesions
- Dual model support: Apple Specialist (4 classes) & PlantVillage Full (38 classes)

Run locally:
    pip install gradio
    python app.py
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

try:
    import gradio as gr
except ImportError:
    raise ImportError("Gradio is not installed. Please run: pip install gradio")

from src.models.classifier import LeafClassifier
from src.interpretability.gradcam import GradCAM, get_last_conv_layer
from src.interpretability.pseudo_boxes import cam_to_bbox, denormalize
from src.data.dataset import default_transforms, APPLE_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = default_transforms(train=False, image_size=224)

# Candidate checkpoint search paths (checks workspace first, then user's runs directory)
SEARCH_DIRS = [
    Path("runs"),
    Path.home() / "runs" / "plant_leaf",
    Path("/content/drive/MyDrive/apple_leaf_runs"),
]


def find_checkpoint(subfolder: str) -> Optional[Path]:
    for base in SEARCH_DIRS:
        candidate = base / subfolder / "best_model.pt"
        if candidate.exists():
            return candidate
    return None


MODEL_CONFIGS = {
    "Apple Specialist (4 Classes)": {
        "subfolder": "baseline_apple_only",
        "classes": ["Apple Scab", "Black Rot", "Cedar Rust", "Healthy"],
        "num_classes": 4,
    },
    "Full PlantVillage (38 Classes)": {
        "subfolder": "baseline_full",
        "num_classes": 38,
    },
}

# Cache loaded models in memory
_MODEL_CACHE: Dict[str, Tuple[LeafClassifier, list]] = {}


def get_model(model_name: str) -> Tuple[Optional[LeafClassifier], list]:
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    cfg = MODEL_CONFIGS[model_name]
    ckpt_path = find_checkpoint(cfg["subfolder"])
    if not ckpt_path:
        return None, []

    # Resolve class names
    classes = cfg.get("classes")
    if not classes:
        classes_file = ckpt_path.parent / "classes.json"
        if classes_file.exists():
            classes = [c.replace("Apple___", "").replace("_", " ") for c in json.loads(classes_file.read_text())]
        else:
            classes = [f"Class {i}" for i in range(cfg["num_classes"])]

    model = LeafClassifier(num_classes=len(classes), backbone="efficientnet_b0", pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    _MODEL_CACHE[model_name] = (model, classes)
    return model, classes


def analyze_leaf(
    image: np.ndarray,
    model_name: str,
    cam_threshold: float,
) -> Tuple[Optional[Dict[str, float]], Optional[np.ndarray], str]:
    if image is None:
        return None, None, "Please upload or select a leaf image."

    model, classes = get_model(model_name)
    if model is None:
        return None, None, f"⚠️ Checkpoint for '{model_name}' not found. Please train the model first or place best_model.pt in runs/."

    # Preprocess
    pil_img = Image.fromarray(image).convert("RGB")
    tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    # Classify
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_k = min(5, len(classes))
    top_probs, top_indices = torch.topk(probs, k=top_k)
    confidences = {classes[idx]: float(prob) for idx, prob in zip(top_indices, top_probs)}

    # Grad-CAM on the top predicted disease
    top_class_idx = top_indices[0].item()
    target_layer = get_last_conv_layer(model)
    gradcam = GradCAM(model, target_layer)
    cam, _ = gradcam(tensor, class_idx=top_class_idx)

    # Compose overlay
    cam_np = cam.cpu().numpy()
    img_resized = np.array(pil_img.resize((224, 224)))
    colormap = plt.cm.jet(cam_np)[:, :, :3]
    heatmap_rgb = (colormap * 255).astype(np.uint8)
    overlay = (0.50 * heatmap_rgb + 0.50 * img_resized).astype(np.uint8)

    # Weakly supervised bounding box around lesion
    bbox = cam_to_bbox(cam.cpu(), threshold=cam_threshold)
    status_msg = f"Top Diagnosis: **{classes[top_class_idx]}** ({top_probs[0].item() * 100:.1f}% confidence)"
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (34, 197, 94), 2)
        cv2.putText(
            overlay,
            "Lesion",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (34, 197, 94),
            2,
            cv2.LINE_AA,
        )
        status_msg += f" | Lesion localized at [{x}, {y}, {w}, {h}]"
    else:
        status_msg += " | No high-activation lesion region detected above threshold."

    return confidences, overlay, status_msg


# Built-in sample images for 1-click testing
SAMPLE_IMAGES = []
samples_dir = Path("samples")
if samples_dir.exists():
    for sample_file in sorted(samples_dir.glob("*.jpg")):
        SAMPLE_IMAGES.append([str(sample_file), "Apple Specialist (4 Classes)", 0.5])


# Build Gradio UI
with gr.Blocks(theme=gr.themes.Soft(), title="Leaf Disease Detection & Localization") as demo:
    gr.Markdown(
        """
        # 🌿 Leaf Disease Detection & Weakly-Supervised Localization
        Upload an image of a diseased or healthy leaf to diagnose pathology, inspect **Grad-CAM attention heatmaps**, and localize lesions without manual bounding boxes.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Leaf Photo (or use webcam)")
            model_selector = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value="Apple Specialist (4 Classes)",
                label="Classification Model",
            )
            threshold_slider = gr.Slider(
                minimum=0.2,
                maximum=0.8,
                value=0.5,
                step=0.05,
                label="Grad-CAM Lesion Detection Threshold",
            )
            analyze_btn = gr.Button("🔍 Diagnose Leaf & Localize Lesion", variant="primary")

        with gr.Column(scale=1):
            status_text = gr.Markdown("Ready to analyze leaf.")
            predictions_output = gr.Label(num_top_classes=5, label="Predicted Diseases (Confidence)")
            visualization_output = gr.Image(type="numpy", label="Grad-CAM Lesion Heatmap & Bounding Box")

    analyze_btn.click(
        fn=analyze_leaf,
        inputs=[image_input, model_selector, threshold_slider],
        outputs=[predictions_output, visualization_output, status_text],
    )

    if SAMPLE_IMAGES:
        gr.Examples(
            examples=SAMPLE_IMAGES,
            inputs=[image_input, model_selector, threshold_slider],
            outputs=[predictions_output, visualization_output, status_text],
            fn=analyze_leaf,
            cache_examples=False,
            label="1-Click Sample Leaves (Apple Dataset)",
        )

if __name__ == "__main__":
    demo.launch()

