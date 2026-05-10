import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Channel Attention
# =========================================================
class ChannelAttention(nn.Module):

    def __init__(self, in_planes, reduction=8):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(
                in_planes,
                in_planes // reduction,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_planes // reduction,
                in_planes,
                kernel_size=1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc(
            self.avg_pool(x)
        )

        max_out = self.fc(
            self.max_pool(x)
        )

        out = avg_out + max_out

        return self.sigmoid(out)


# =========================================================
# Spatial Attention
# =========================================================
class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # 平均池化
        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        # 最大池化
        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        x = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        x = self.conv(x)

        return self.sigmoid(x)


# =========================================================
# CBAM Block
# =========================================================
class CBAM(nn.Module):

    def __init__(self, channels, reduction=8):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(
            channels,
            reduction
        )

        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        # =================================================
        # Channel Attention
        # =================================================
        ca = self.channel_attention(x)

        x = x * ca

        # =================================================
        # Spatial Attention
        # =================================================
        sa = self.spatial_attention(x)

        x = x * sa

        return x


# =========================================================
# EEGNet + CBAM
# =========================================================
class EEGNet_CBAM(nn.Module):

    def __init__(
            self,
            chans=4,
            time_point=3000,
            nb_classes=5,
            dropoutRate=0.5
    ):
        super(EEGNet_CBAM, self).__init__()

        # First Conv
        self.firstconv = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False
            ),

            nn.BatchNorm2d(8)
        )

        # Depthwise Conv
        self.depthwiseConv = nn.Sequential(

            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(chans, 1),
                groups=8,
                bias=False
            ),

            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )

        # CBAM Attention 1
        self.cbam1 = CBAM(16)

        # Separable Conv
        self.separableConv = nn.Sequential(

            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=16,
                bias=False
            ),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(1, 1),
                bias=False
            ),

            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )

        # CBAM Attention 2
        self.cbam2 = CBAM(32)

        # 自动计算 flatten size
        self.flatten_size = self._get_flatten_size(
            chans,
            time_point
        )

        print("Flatten Size:", self.flatten_size)

        # Classifier
        self.classifier = nn.Sequential(

            nn.Linear(
                self.flatten_size,
                128
            ),

            nn.ReLU(),
            nn.Dropout(dropoutRate),
            nn.Linear(
                128,
                nb_classes
            )
        )

    # =====================================================
    # 自动推导 flatten size
    # =====================================================
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
            x = self.cbam1(x)
            x = self.separableConv(x)
            x = self.cbam2(x)
            flatten_size = x.view(1, -1).shape[1]

        return flatten_size

    def forward(self, x):
        # 输入:[B, C, T]
        x = x.unsqueeze(1)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        # CBAM Attention
        x = self.cbam1(x)

        x = self.separableConv(x)
        # CBAM Attention
        x = self.cbam2(x)

        # Flatten
        x = x.flatten(start_dim=1)

        # Classification
        x = self.classifier(x)
        return x