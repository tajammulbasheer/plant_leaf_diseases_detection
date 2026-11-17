import unittest
import torch

from src.models.classifier import LeafClassifier
from src.models.unet import UNet
from src.models.gan import Generator, Discriminator


class TestModels(unittest.TestCase):
    def test_classifier_forward(self):
        model = LeafClassifier(num_classes=4, backbone="efficientnet_b0", pretrained=False)
        model.eval()
        dummy_x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy_x)
        self.assertEqual(out.shape, (2, 4), "Classifier output logits shape must match (batch_size, num_classes)")

    def test_unet_forward(self):
        model = UNet(in_channels=3, out_channels=1)
        model.eval()
        dummy_x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(dummy_x)
        self.assertEqual(out.shape, (2, 1, 128, 128), "UNet output mask shape must match (batch_size, 1, H, W)")

    def test_gan_forward(self):
        latent_dim = 100
        net_g = Generator(latent_dim=latent_dim, img_channels=3)
        net_d = Discriminator(img_channels=3)
        net_g.eval()
        net_d.eval()

        z = torch.randn(2, latent_dim, 1, 1)
        with torch.no_grad():
            fake_imgs = net_g(z)
            d_out = net_d(fake_imgs)

        self.assertEqual(fake_imgs.shape, (2, 3, 64, 64), "Generator output shape must be (batch_size, 3, 64, 64)")
        self.assertEqual(d_out.shape, (2,), "Discriminator output shape must be (batch_size,)")


if __name__ == "__main__":
    unittest.main()
