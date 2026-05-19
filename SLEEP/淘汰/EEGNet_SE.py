import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# SE Attention Block
# =========================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),

            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape:[B, C, H, W]
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# =========================================================
# EEGNet + SE Attention
# =========================================================
class EEGNet_SE(nn.Module):

    def __init__(
            self,
            chans=4,
            time_point=3000,
            nb_classes=5,
            dropoutRate=0.5
    ):
        super(EEGNet_SE, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(
                1,
                8,
                kernel_size=(1, 64),
                stride=1,
                padding=(0, 32),
                bias=False
            ),
            nn.BatchNorm2d(8)
        )

        # Depthwise Conv
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
                8,
                16,
                kernel_size=(chans, 1),
                groups=8,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )

        # SE Attention（关键升级）
        self.se1 = SEBlock(16)

        # Separable Conv
        self.separableConv = nn.Sequential(
            nn.Conv2d(
                16,
                16,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=16,
                bias=False
            ),

            nn.Conv2d(
                16,
                32,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )

        # 第二层 SE Attention
        self.se2 = SEBlock(32)

        # 自动计算 flatten size
        self.flatten_size = self._get_flatten_size(
            chans,
            time_point
        )

        # Classifier
        self.classify = nn.Sequential(
            nn.Linear(
                self.flatten_size,
                128
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(
                128,
                nb_classes
            )
        )

    # 自动推导 Linear 输入维度
    def _get_flatten_size(self, chans, time_point):

        with torch.no_grad():
            x = torch.zeros(
                1,
                1,
                chans,
                time_point
            )

            x = self.firstconv(x)
            x = self.depthwiseConv(x)
            x = self.se1(x)
            x = self.separableConv(x)
            x = self.se2(x)

            flatten_size = x.view(1, -1).shape[1]
        return flatten_size


    def forward(self, x):
        # 输入:[B, C, T]
        x = x.unsqueeze(1)
        # -> [B, 1, C, T]
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        # SE Attention
        x = self.se1(x)
        x = self.separableConv(x)
        # SE Attention
        x = self.se2(x)
        x = x.flatten(start_dim=1)
        x = self.classify(x)
        return x