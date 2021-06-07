import torch
import torch.nn as nn
from math import ceil


BASE_MODEL = [
    # [expand_ratio, C_i (channels), L_i (layers), stride, kernel_size]
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

PHI_VALUES = {
    # F_VALUES["version"]:(phi, resolution, drop_rate)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class ConvBlock(nn.Module):
    """A standart convolution/depthwiseconv block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        """
        :param in_channels: ``int``, in channels
        :param out_channels: ``int``, out_channels
        :param kernel_size: ``int or Tuple(int, int)``, kernel size
        :param stride: ``int or Tuple(int, int)``, stride
        :param padding: ``int or Tuple(int, int)``, padding
        :param groups: ``int``, if groups=1 then block will be standart convolutional, if groups=in_channels
        block will be depthwise convolutional
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        :param x: ``Tensor([N, C, H, W])``
        :return: ``Tensor([N, C, H, W])``
        """
        return self.act(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    """SE block"""
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block of MobileNet"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio,
                 reduction=4, survival_prob=0.8):
        """
        :param in_channels: ``int``, in channels
        :param out_channels: ``int``, out_channels
        :param kernel_size: ``int or Tuple(int, int)``, kernel size
        :param stride: ``int or Tuple(int, int)``, stride
        :param padding: ``int or Tuple(int, int)``, padding
        :param expand_ratio: ``int``, expand ratio
        :param reduction: ``int``, for squeeze excitation
        :param survival_prob: ``float``, probability for stochastic depth
        """
        super().__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = ConvBlock(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    """Efficient"""
    def __init__(self, verison, num_classes):
        """
        :param verison: ``str``, version of factor should be [``b_0``, ..., ``b_k``], where k in {0,...,7}
        :param num_classes: ``int``, num classes of classifier
        """
        super().__init__()
        depth_factor, width_factor, drop_rate = self.calculate_factors(verison)
        last_channels = ceil(1280 * width_factor)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(last_channels, num_classes)
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        """
        Calculate factors
        :param version: ``str`` current version
        :param alpha: ``float``, const, depth, in the paper default=1.2
        :param beta: ``float``, const, width, in the paper default=1.1
        :return: ``Tuple()``, factors
        """
        phi, gamma, drop_rate = PHI_VALUES[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return depth_factor, width_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        """
        Creates features
        :param width_factor: ``float``, width factor
        :param depth_factor: ``float``, depth factor
        :param last_channels: ``int``, num output channels for last conv block
        :return: ``nn.Sequential``
        """
        channels = int(32 * width_factor)
        features = [ConvBlock(3, channels, kernel_size=3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in BASE_MODEL:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    )
                )
                in_channels = out_channels

        features.append(
            ConvBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)

    def forward(self, x):
        """
        :param x: ``Tensor([N, C, H, W])``
        :return: ``Tensor([N, NUM_CLASSES])``
        """
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))
