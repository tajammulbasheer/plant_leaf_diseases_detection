import unittest
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.utils.metrics import evaluate_classifier
from src.incremental.metrics import compute_forgetting, average_incremental_accuracy


class TestMetrics(unittest.TestCase):
    def test_evaluate_classifier(self):
        # Mock model that simply returns input as logits
        class IdentityModel(torch.nn.Module):
            def forward(self, x):
                return x

        model = IdentityModel()
        # 4 samples: 2 for class 0, 2 for class 1
        logits = torch.tensor([
            [2.0, 0.0],
            [3.0, 1.0],
            [0.5, 2.5],
            [1.0, 0.0], # intentionally misclassified (predicted 0, label 1)
        ])
        labels = torch.tensor([0, 0, 1, 1])
        ds = TensorDataset(logits, labels)
        dl = DataLoader(ds, batch_size=2)

        results = evaluate_classifier(model, dl, device=torch.device("cpu"), class_names=["A", "B"])
        self.assertAlmostEqual(results["accuracy"], 0.75, places=2)
        self.assertIn("macro_f1", results)
        self.assertIn("macro_precision", results)
        self.assertIn("macro_recall", results)

    def test_incremental_metrics(self):
        # 3 tasks:
        # After T1: T1 acc = 0.90
        # After T2: T1 acc = 0.70, T2 acc = 0.80
        # After T3: T1 acc = 0.40, T2 acc = 0.60, T3 acc = 0.85
        matrix = np.array([
            [0.90, np.nan, np.nan],
            [0.70, 0.80,   np.nan],
            [0.40, 0.60,   0.85],
        ])

        # Forgetting for T1 = 0.90 - 0.40 = 0.50
        # Forgetting for T2 = 0.80 - 0.60 = 0.20
        # Average forgetting = (0.50 + 0.20) / 2 = 0.35
        forgetting = compute_forgetting(matrix)
        self.assertAlmostEqual(forgetting, 0.35, places=3)

        # Average final accuracy = (0.40 + 0.60 + 0.85) / 3 = 0.6167
        avg_acc = average_incremental_accuracy(matrix)
        self.assertAlmostEqual(avg_acc, 0.6167, places=3)


if __name__ == "__main__":
    unittest.main()

