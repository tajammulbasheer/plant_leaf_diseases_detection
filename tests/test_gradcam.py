import unittest
import torch
import numpy as np

from src.models.classifier import LeafClassifier
from src.interpretability.gradcam import GradCAM, get_last_conv_layer
from src.interpretability.pseudo_boxes import cam_to_bbox, denormalize


class TestGradCAM(unittest.TestCase):
    def test_gradcam_and_bbox(self):
        model = LeafClassifier(num_classes=4, backbone="efficientnet_b0", pretrained=False)
        model.eval()

        target_layer = get_last_conv_layer(model)
        self.assertIsNotNone(target_layer, "Must identify last convolutional layer")

        gradcam = GradCAM(model, target_layer)
        dummy_x = torch.randn(1, 3, 224, 224, requires_grad=True)

        cam, pred_idx = gradcam(dummy_x, class_idx=0)
        self.assertEqual(cam.shape, (224, 224), "Grad-CAM heatmap shape must match (H, W)")
        self.assertGreaterEqual(cam.min().item(), 0.0)
        self.assertLessEqual(cam.max().item(), 1.0)

        # Test cam_to_bbox on synthetic activation mask
        synthetic_cam = torch.zeros(224, 224)
        # Put high activation patch from y=50..100, x=40..90
        synthetic_cam[50:100, 40:90] = 0.9

        bbox = cam_to_bbox(synthetic_cam, threshold=0.5)
        self.assertIsNotNone(bbox)
        x, y, w, h = bbox
        self.assertEqual(x, 40)
        self.assertEqual(y, 50)
        self.assertEqual(w, 50)
        self.assertEqual(h, 50)

    def test_denormalize(self):
        tensor = torch.zeros(3, 224, 224)
        denorm = denormalize(tensor)
        self.assertEqual(denorm.shape, (3, 224, 224))
        self.assertGreaterEqual(denorm.min().item(), 0.0)
        self.assertLessEqual(denorm.max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()

