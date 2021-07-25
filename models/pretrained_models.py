import timm
import torch.nn as nn
from config import cfg


class WideResnet50(nn.Module):
    """WideResnet50"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('wide_resnet50_2', pretrained=pretrained, in_chans=1)
        self.n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.n_features, cfg.NUM_CLASSES)

    def forward(self, x):
        return self.model(x)


class EfficientNetP(nn.Module):
    """EfficientNet b0-b7"""
    def __init__(self, version, pretrained=True):
        super().__init__()
        assert version in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']
        self.model = timm.create_model(f'efficientnet_{version}', pretrained=pretrained, in_chans=1)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, cfg.NUM_CLASSES)

    def forward(self, x):
        return self.model(x)


class NFNETL0(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('nfnet_l0', pretrained=pretrained, in_chans=1)
        self.n_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(self.n_features, cfg.NUM_CLASSES)

    def forward(self, x):
        return self.model(x)


def get_model(model_name, version='b0', pretrained=True):
    if model_name == 'efficient':
        return EfficientNetP(version, pretrained=pretrained)
    elif model_name == 'wide_resnet50_2':
        return WideResnet50(pretrained=pretrained)
    elif model_name == 'nf_net':
        return NFNETL0(pretrained=pretrained)
