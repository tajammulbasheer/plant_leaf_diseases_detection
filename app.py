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

Checkpoints are looked up as <dir>/baseline_apple_only/best_model.pt and <dir>/baseline_full/best_model.pt,
with <dir> taken from $RUN_DIR first, then the defaults in SEARCH_DIRS.
"""

import os
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
    gr = None

try:  # Hugging Face ZeroGPU Spaces attach a GPU only while a @spaces.GPU function runs
    import spaces
    gpu = spaces.GPU
except ImportError:  # local runs and CPU Spaces: no-op
    def gpu(fn):
        return fn

from src.models.classifier import LeafClassifier, load_checkpoint_classes
from src.interpretability.gradcam import GradCAM, get_last_conv_layer
from src.interpretability.pseudo_boxes import cam_to_bbox, denormalize
from src.data.dataset import default_transforms, APPLE_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = default_transforms(train=False, image_size=224)
ROOT = Path(__file__).resolve().parent

# Candidate checkpoint search paths: $RUN_DIR, the repo's runs/, the local and Colab defaults
SEARCH_DIRS = [Path(os.environ["RUN_DIR"]).expanduser()] if os.environ.get("RUN_DIR") else []
SEARCH_DIRS += [
    ROOT / "runs",
    Path.home() / "runs" / "plant_leaf",
    Path("/content/drive/MyDrive/apple_leaf_runs"),
]


def find_checkpoint(subfolder: str) -> Optional[Path]:
    for base in SEARCH_DIRS:
        try:
            candidate = base / subfolder / "best_model.pt"
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


MODEL_CONFIGS = {
    "Apple Specialist (4 Classes)": {
        "subfolder": "baseline_apple_only",
        "expected_classes": APPLE_CLASSES,
    },
    "Full PlantVillage (38 Classes)": {
        "subfolder": "baseline_full",
        "expected_classes": None,
    },
}

# Cache loaded models (with their Grad-CAM, so its hooks are registered once per model, not per request)
_MODEL_CACHE: Dict[str, Tuple[LeafClassifier, list, GradCAM]] = {}


def display_name(folder_name: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato: Early blight', 'Pepper,_bell___healthy' -> 'Pepper, bell: Healthy'."""
    crop, _, disease = folder_name.partition("___")
    crop, disease = crop.replace("_", " ").strip(), disease.replace("_", " ").strip()
    return f"{crop}: {disease[:1].upper()}{disease[1:]}" if disease else crop


def get_model(model_name: str) -> Tuple[Optional[LeafClassifier], list, Optional[GradCAM]]:
    """Returns (model, raw class names from classes.json, GradCAM), or (None, [], None) if not found."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    cfg = MODEL_CONFIGS[model_name]
    ckpt_path = find_checkpoint(cfg["subfolder"])
    if not ckpt_path:
        return None, [], None

    # classes.json is written by train_classifier next to the checkpoint, in label-index order
    classes = load_checkpoint_classes(ckpt_path, cfg["expected_classes"])
    if not classes:
        raise FileNotFoundError(f"{ckpt_path.parent / 'classes.json'} is missing; cannot map outputs to classes.")

    model = LeafClassifier(num_classes=len(classes), backbone="efficientnet_b0", pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    # Gradio runs each event one request at a time by default, so sharing one GradCAM is safe.
    gradcam = GradCAM(model, get_last_conv_layer(model))

    _MODEL_CACHE[model_name] = (model, classes, gradcam)
    return model, classes, gradcam


@gpu
def analyze_leaf(
    image: np.ndarray,
    model_name: str,
    cam_threshold: float,
) -> Tuple[Optional[Dict[str, float]], Optional[np.ndarray], str]:
    if image is None:
        return None, None, "Please upload or select a leaf image."

    model, classes, gradcam = get_model(model_name)
    if model is None:
        subfolder = MODEL_CONFIGS[model_name]["subfolder"]
        return None, None, (f"⚠️ Checkpoint for '{model_name}' not found. Train the model first, or set RUN_DIR "
                            f"to the folder containing {subfolder}/best_model.pt.")

    # Preprocess
    pil_img = Image.fromarray(image).convert("RGB")
    tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    # Classify
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_k = min(5, len(classes))
    top_probs, top_indices = torch.topk(probs, k=top_k)
    confidences = {display_name(classes[idx]): float(prob) for idx, prob in zip(top_indices, top_probs)}

    # Grad-CAM on the top predicted class
    top_class_idx = top_indices[0].item()
    is_healthy = classes[top_class_idx].lower().endswith("healthy")
    cam, _ = gradcam(tensor, class_idx=top_class_idx)

    # Compose overlay
    cam_np = cam.cpu().numpy()
    img_resized = np.array(pil_img.resize((224, 224)))
    colormap = plt.cm.jet(cam_np)[:, :, :3]
    heatmap_rgb = (colormap * 255).astype(np.uint8)
    overlay = (0.50 * heatmap_rgb + 0.50 * img_resized).astype(np.uint8)

    # Weakly supervised bounding box around the most activated region. For a healthy prediction
    # there is no lesion to localize: the box only shows where the model looked.
    bbox = cam_to_bbox(cam.cpu(), threshold=cam_threshold)
    box_label = "Model attention" if is_healthy else "Lesion"
    status_msg = f"Top Diagnosis: **{display_name(classes[top_class_idx])}** ({top_probs[0].item() * 100:.1f}% confidence)"
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (34, 197, 94), 2)
        cv2.putText(
            overlay,
            box_label,
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (34, 197, 94),
            2,
            cv2.LINE_AA,
        )
        if is_healthy:
            status_msg += f" | No lesion: the box at [{x}, {y}, {w}, {h}] only marks where the model looked"
        else:
            status_msg += f" | Lesion localized at [{x}, {y}, {w}, {h}]"
    else:
        status_msg += " | No high-activation region detected above threshold."

    return confidences, overlay, status_msg


# Built-in sample images for 1-click testing
SAMPLE_IMAGES = []
samples_dir = ROOT / "samples"
if samples_dir.exists():
    for sample_file in sorted(samples_dir.glob("*.jpg")):
        SAMPLE_IMAGES.append([str(sample_file), "Apple Specialist (4 Classes)", 0.5])


def build_demo():
    if gr is None:
        raise ImportError("Gradio is not installed. Please run: pip install gradio")

    with gr.Blocks(title="Leaf Disease Detection & Localization") as demo:
        gr.Markdown(
            """
            # 🌿 Leaf Disease Detection & Weakly-Supervised Localization
            Upload an image of a diseased or healthy leaf to diagnose pathology, inspect **Grad-CAM attention heatmaps**, and localize lesions without manual bounding boxes.

            **Limitation:** the models were trained on PlantVillage lab photos (one leaf, plain background).
            On real field photos (PlantDoc) the Apple model's accuracy drops to about half; see the
            cross-dataset results in the README. Expect unreliable predictions on cluttered photos.
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

    return demo


if __name__ == "__main__":
    if gr is None:
        print("Error: Gradio is not installed. Please run: pip install gradio")
        exit(1)
    for name in MODEL_CONFIGS:  # load at startup (ZeroGPU expects models placed before the first request)
        get_model(name)
    demo = build_demo()
    try:
        demo.launch(theme=gr.themes.Soft())
    except TypeError:
        demo.launch()

