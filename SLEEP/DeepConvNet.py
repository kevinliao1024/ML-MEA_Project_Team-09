import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# DeepConvNet for EEG / Sleep Staging
# =========================================================
class DeepConvNet(nn.Module):

    def __init__(
            self,
            chans=4,
            time_point=3000,
            nb_classes=5,
            dropoutRate=0.5
    ):
        super(DeepConvNet, self).__init__()

        self.block1 = nn.Sequential(

            # Temporal Conv
            nn.Conv2d(
                in_channels=1,
                out_channels=25,
                kernel_size=(1, 10),
                stride=1,
                padding=(0, 5),
                bias=False
            ),

            # Spatial Conv
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(chans, 1),
                stride=1,
                bias=False
            ),

            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(
                kernel_size=(1, 3),
                stride=(1, 3)
            ),

            nn.Dropout(dropoutRate)
        )

        self.block2 = nn.Sequential(

            nn.Conv2d(
                in_channels=25,
                out_channels=50,
                kernel_size=(1, 10),
                padding=(0, 5),
                bias=False
            ),

            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(
                kernel_size=(1, 3),
                stride=(1, 3)
            ),

            nn.Dropout(dropoutRate)
        )

        self.block3 = nn.Sequential(

            nn.Conv2d(
                in_channels=50,
                out_channels=100,
                kernel_size=(1, 10),
                padding=(0, 5),
                bias=False
            ),

            nn.BatchNorm2d(100),
            nn.ELU(),
            
            nn.MaxPool2d(
                kernel_size=(1, 3),
                stride=(1, 3)
            ),

            nn.Dropout(dropoutRate)
        )

        self.block4 = nn.Sequential(

            nn.Conv2d(
                in_channels=100,
                out_channels=200,
                kernel_size=(1, 10),
                padding=(0, 5),
                bias=False
            ),

            nn.BatchNorm2d(200),
            nn.ELU(),

            nn.MaxPool2d(
                kernel_size=(1, 3),
                stride=(1, 3)
            ),

            nn.Dropout(dropoutRate)
        )

        # =================================================
        # 自动计算 flatten size
        # =================================================
        self.flatten_size = self._get_flatten_size(
            chans,
            time_point
        )

        print("Flatten Size:", self.flatten_size)

        # =================================================
        # Classifier
        # =================================================
        self.classifier = nn.Sequential(

            nn.Linear(
                self.flatten_size,
                256
            ),

            nn.ReLU(),
            nn.Dropout(dropoutRate),

            nn.Linear(
                256,
                nb_classes
            )
        )

    # =====================================================
    # 自动计算 flatten size
    # =====================================================
    def _get_flatten_size(self, chans, time_point):
        with torch.no_grad():
            x = torch.zeros(
                1,
                1,
                chans,
                time_point
            )

            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)

            flatten_size = x.view(1, -1).shape[1]

        return flatten_size

    def forward(self, x):

        # 输入:[B, C, T]

        x = x.unsqueeze(1)

        # [B, 1, C, T]

        # DeepConvNet Blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Flatten
        x = x.flatten(start_dim=1)

        # Classification
        x = self.classifier(x)

        return x