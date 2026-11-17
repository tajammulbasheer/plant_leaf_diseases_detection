import tempfile
import unittest
from pathlib import Path
from PIL import Image
import torch

from src.data.dataset import default_transforms, LeafDataset, APPLE_CLASSES, IMAGENET_MEAN, IMAGENET_STD


class TestDataset(unittest.TestCase):
    def test_default_transforms_eval(self):
        tf = default_transforms(image_size=224, train=False)
        img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        tensor = tf(img)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 224, 224), "Eval transform must produce (3, 224, 224) tensor")

    def test_default_transforms_train(self):
        tf = default_transforms(image_size=128, train=True)
        img = Image.new("RGB", (200, 200), color=(50, 100, 150))
        tensor = tf(img)
        self.assertEqual(tensor.shape, (3, 128, 128), "Train transform must resize to given image_size")

    def test_leaf_dataset_and_few_shot_sampling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classes = ["ClassA", "ClassB"]
            # Create 10 dummy images in each class folder
            for c in classes:
                c_dir = root / c
                c_dir.mkdir()
                for i in range(10):
                    dummy_img = Image.new("RGB", (32, 32), color=(i * 20, 50, 100))
                    dummy_img.save(c_dir / f"img_{i}.jpg")

            # 1. Full dataset
            full_ds = LeafDataset(root=str(root), transform=default_transforms(image_size=64, train=False))
            self.assertEqual(len(full_ds), 20, "Total samples should be 20")

            # 2. Few-shot (k=3 per class -> total 6)
            few_ds = LeafDataset(root=str(root), k_shot=3, seed=42, transform=default_transforms(image_size=64, train=False))
            self.assertEqual(len(few_ds), 6, "3-shot dataset with 2 classes must contain exactly 6 samples")

            item, label = few_ds[0]
            self.assertEqual(item.shape, (3, 64, 64))
            self.assertIn(label, [0, 1])


if __name__ == "__main__":
    unittest.main()

