import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform

class FCOSMobileNetMapped(nn.Module):
    def __init__(self, input_channels, num_classes, image_size=(400, 400), pretrained=True):
        super().__init__()

        self.channel_mapper = nn.Sequential(
            nn.Conv2d(input_channels, 3, kernel_size=1),
            nn.ReLU()
        )

        # Initialize FCOS base model
        self.detector = resnet_fpn_backbone('resnet18', pretrained=True)

        # Update the number of output classes
        in_features = self.detector.head.classification_head.conv[0].in_channels
        num_anchors = self.detector.head.classification_head.num_anchors
        self.detector.head.classification_head.num_classes = num_classes
        self.detector.head.classification_head.cls_logits = nn.Conv2d(
            in_features, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )

        # Adjust transform for input normalization and resizing
        self.detector.transform = GeneralizedRCNNTransform(
            min_size=image_size[0],
            max_size=image_size[1],
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0]
        )

    def forward(self, x, targets=None):
        # Normalize each sample individually
        x = [(img - img.mean()) / (img.std() + 1e-6) for img in x]
        x = [self.channel_mapper(img) for img in x]

        if self.training:
            return self.detector(x, targets)
        else:
            return self.detector(x)
