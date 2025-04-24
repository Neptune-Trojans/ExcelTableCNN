import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch.nn as nn

class FasterRCNNMobileNetMapped2(nn.Module):
    def __init__(self, input_channels, num_classes, image_size=(256, 256), pretrained=True):
        super().__init__()

        # 1x1 conv to reduce input channels to 3
        self.channel_mapper = nn.Conv2d(input_channels, 3, kernel_size=1)

        # Backbone (without FPN wrapper here)
        backbone = mobilenet_backbone('mobilenet_v3_large', pretrained=pretrained, fpn=True)

        # Construct detector manually to override transform
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2)

        self.detector = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            transform=GeneralizedRCNNTransform(
                min_size=image_size[0],
                max_size=image_size[1],
                image_mean=[0.0, 0.0, 0.0],
                image_std=[1.0, 1.0, 1.0]
            )
        )

        # Replace box predictor (classification head)
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x, targets=None):
        # Map 17-channel input to 3-channel for MobileNet
        x = [self.channel_mapper(img) for img in x]

        if self.training:
            return self.detector(x, targets)
        else:
            return self.detector(x)
