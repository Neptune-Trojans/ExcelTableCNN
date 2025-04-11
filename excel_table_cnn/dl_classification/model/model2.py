import torch
import torch.nn as nn
from torchvision.models.detection import retinanet_resnet50_fpn

from excel_table_cnn.dl_classification.tensors import DataframeTensors
from excel_table_cnn.train_test_helpers import get_table_features


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

    def forward(self, images, targets=None):
        # images: List[Tensor] with shape [C, H, W] each
        processed = [self.channel_mapper(img.unsqueeze(0)).squeeze(0) for img in images]
        if self.training:
            return self.detector(processed, targets)
        else:
            return self.detector(processed)



if __name__ == '__main__':
    # Works fine

    images = [torch.randn(12, 512, 512, device='mps')]

    targets = [{
        'boxes': torch.tensor([[0., 0., 27., 13.]], dtype=torch.float32, device='mps'),
        'labels': torch.tensor([1], dtype=torch.int64, device='mps')
    }]

    model = MultiChannelRetinaNet(input_channels=12, num_classes=2).to('mps')
    model.train()
    losses = model(images, targets)
    print(losses)