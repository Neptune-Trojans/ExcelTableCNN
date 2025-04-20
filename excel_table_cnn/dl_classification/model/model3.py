import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

class FasterRCNNMobileNetMapped(nn.Module):
    def __init__(self, input_channels, num_classes, pretrained=True):
        super().__init__()

        # 1x1 conv to reduce input channels to 3 (for MobileNetV3)
        self.channel_mapper = nn.Conv2d(input_channels, 3, kernel_size=1)

        # Load Faster R-CNN with MobileNetV3 backbone
        self.detector = fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained)

        # Modify the classification head for custom number of classes
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = nn.Sequential(
            nn.Linear(in_features, num_classes + 1),  # +1 for background
            nn.Linear(in_features, num_classes + 1)   # optional second head if needed
        )

        # OR: use TorchVision's built-in predictor replacement
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x, targets=None):
        # x is a list of tensors; apply channel_mapper to each one
        x = [self.channel_mapper(img) for img in x]

        if self.training:
            return self.detector(x, targets)
        else:
            return self.detector(x)


if __name__ == '__main__':
    # Works fine

    images = [torch.randn(12, 512, 512, device='mps')]

    targets = [{
        'boxes': torch.tensor([[0., 0., 27., 13.]], dtype=torch.float32, device='mps'),
        'labels': torch.tensor([1], dtype=torch.int64, device='mps')
    }]

    model = FasterRCNNMobileNetMapped(input_channels=12, num_classes=2).to('mps')
    model.train()
    losses = model(images, targets)
    print(losses)