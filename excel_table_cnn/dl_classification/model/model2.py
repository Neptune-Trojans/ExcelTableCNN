import torch
import torch.nn as nn
from torchvision.models.detection import retinanet_resnet50_fpn


class MultiChannelRetinaNet(nn.Module):
    def __init__(self, input_channels: int = 12, num_classes: int = 91, pretrained=True):
        super().__init__()

        # 1x1 conv to reduce input channels to 3
        self.channel_mapper = nn.Conv2d(input_channels, 3, kernel_size=1)

        # Load RetinaNet with ResNet50 backbone
        self.detector = retinanet_resnet50_fpn(pretrained=pretrained)

        # Modify the number of classes if needed
        in_features = self.detector.head.classification_head.num_anchors
        self.detector.head.classification_head.num_classes = num_classes
        self.detector.head.classification_head.cls_logits = nn.Conv2d(
            self.detector.head.classification_head.conv[0].out_channels,
            in_features * num_classes,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x, targets=None):
        x = self.channel_mapper(x)  # Convert to 3 channels
        if self.training:
            return self.detector(x, targets)
        else:
            return self.detector(x)



if __name__ == '__main__':
    # Works fine

    img = torch.randn(1, 12, 20000, 800)  # 640x480

    model = MultiChannelRetinaNet(input_channels=12, num_classes=2, pretrained=False)
    model.eval()


    # Run inference
    with torch.no_grad():
        outputs = model(img)

    print(outputs)