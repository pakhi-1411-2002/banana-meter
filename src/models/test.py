import torch
from multitask_resnet import BananaMultiTaskResNet

model = BananaMultiTaskResNet(
    num_classes=5,
    pretrained=False,
    freeze_backbone=True,
)

x = torch.randn(4, 3, 224, 224)
stage_logits, days_left = model(x)

print("stage_logits shape:", stage_logits.shape)  # should be [4, 5]
print("days_left shape:", days_left.shape)        # should be [4]