import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
        use_whole_features: bool,
    ):
        super().__init__()
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        if use_whole_features:
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else:
            self.body = backbone
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs


class ResnetBackbone(BackboneBase):
    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
        use_whole_features: bool,
    ):
        backbone = getattr(torchvision.models, name)
        backbone = backbone(weights='IMAGENET1K_V1')
        if use_whole_features:
            pass
        else:
            backbone.fc = torch.nn.Identity()
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(
            backbone, train_backbone, num_channels, return_interm_layers, use_whole_features
        )
