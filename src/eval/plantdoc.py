"""
Loader for the PlantDoc dataset, used purely as an out-of-distribution
(OOD) test set to measure how well a PlantVillage-trained model generalizes
to real-world field photos (messy backgrounds, variable lighting, occlusion)
vs. the lab-condition photos PlantVillage is made of.

PlantDoc source: https://github.com/pratikkayal/PlantDoc-Dataset
(Singh et al., "PlantDoc: A Dataset for Visual Plant Disease Detection", 2020)

PlantDoc's apple-relevant classes don't line up 1:1 with PlantVillage's
naming, and it has no direct equivalent for "Black_rot". We evaluate only
on the classes that DO overlap, and report that limitation explicitly
rather than silently dropping it.
"""

from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset
from PIL import Image

# Maps PlantDoc's apple-specific folder names -> PlantVillage's Apple___* labels.
# "Black_rot" is intentionally absent: PlantDoc has no matching apple black rot class.
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Apple leaf": "Apple___healthy",
}

# Complete mapping for all 27 overlapping classes across crops between PlantDoc and PlantVillage (38 classes)
PLANTDOC_FULL_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Apple leaf": "Apple___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
    "Peach leaf": "Peach___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf": "Tomato___healthy",
}


class PlantDocDataset(Dataset):
    """
    Expects PlantDoc's raw folder structure:
        root/<class_name>/*.jpg

    Only folders present in `mapping` are used; non-matching folders are skipped.
    """

    def __init__(
        self,
        root: str,
        class_to_idx: dict,
        mapping: Optional[dict] = None,
        transform: Optional[object] = None,
    ):
        self.root = Path(root)
        self.mapping = mapping or PLANTDOC_FULL_TO_PLANTVILLAGE
        self.transform = transform
        self.samples = []
        self.skipped_classes = []

        for pd_name, pv_name in self.mapping.items():
            if pv_name not in class_to_idx:
                continue
            class_dir = self.root / pd_name
            if not class_dir.exists():
                self.skipped_classes.append(pd_name)
                continue
            label = class_to_idx[pv_name]
            for f in class_dir.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(f), label))

        if not self.samples:
            raise RuntimeError(
                f"No matching PlantDoc classes found under {self.root}. "
                f"Expected one of {list(self.mapping.keys())}."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def missing_classes_note(self) -> str:
        note = f"Evaluated {len(self.samples)} images across {len(self.mapping)} aligned classes."
        if self.skipped_classes:
            note += f" Missing on disk: {self.skipped_classes}."
        return note


class PlantDocAppleDataset(PlantDocDataset):
    """Backward-compatible Apple-only subset (Apple scab, cedar rust, healthy)."""

    def __init__(self, root: str, class_to_idx: dict, transform: Optional[object] = None):
        super().__init__(root=root, class_to_idx=class_to_idx, mapping=PLANTDOC_TO_PLANTVILLAGE, transform=transform)

    @property
    def missing_classes_note(self) -> str:
        note = "Note: 'Black_rot' has no PlantDoc equivalent and is excluded from this OOD eval."
        if self.skipped_classes:
            note += f" Also missing on disk: {self.skipped_classes}."
        return note
