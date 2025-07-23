import torch
import torch.nn as nn
from transformers import DetrForObjectDetection, DetrConfig

class DetrResNet18Mapped(nn.Module):
    def __init__(self, num_classes, image_size=(400, 400), pretrained_backbone=True):
        super().__init__()

        self.input_mapper = nn.Sequential(
            nn.Conv2d(17, 3, kernel_size=1),  # Map from 17 → 3 channels
            nn.ReLU()
        )

        # Load pretrained DETR with ResNet-50 backbone
        self.detr = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_classes,
            num_queries=20,
            ignore_mismatched_sizes=True  # If adjusting label space
        )


    def forward(self, x, targets=None):

        x = self.input_mapper(x)  # → (B, 3, H, W)

        if self.training:
            return self.detr(x, labels=targets)
        else:
            return self.detr(x)



if __name__ == '__main__':
    model = DetrResNet18Mapped(num_classes=2)
    input_tensor = torch.randn(1, 17, 300, 300)
    outputs = model(input_tensor)