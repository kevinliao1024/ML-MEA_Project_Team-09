import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# EEGNet + BiLSTM
# =========================================================
class EEGNet_BiLSTM(nn.Module):

    def __init__(
            self,
            chans=4,
            time_point=3000,
            nb_classes=5,
            dropoutRate=0.5,
            lstm_hidden=64,
            lstm_layers=2
    ):
        super(EEGNet_BiLSTM, self).__init__()

        # =================================================
        # Block 1
        # =================================================
        self.firstconv = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(1, 64),
                stride=1,
                padding=(0, 32),
                bias=False
            ),

            nn.BatchNorm2d(8)
        )

        # =================================================
        # Depthwise Conv
        # =================================================
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

        # =================================================
        # Separable Conv
        # =================================================
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

        # =================================================
        # 自动计算 LSTM 输入维度
        # =================================================
        self.feature_dim, self.sequence_length = self._get_lstm_shape(
            chans,
            time_point
        )

        print("Feature Dim :", self.feature_dim)
        print("Sequence Len:", self.sequence_length)

        # =================================================
        # BiLSTM
        # =================================================
        self.bilstm = nn.LSTM(

            input_size=self.feature_dim,

            hidden_size=lstm_hidden,

            num_layers=lstm_layers,

            batch_first=True,

            bidirectional=True,

            dropout=dropoutRate
        )

        # =================================================
        # Attention Pooling（重要）
        # =================================================
        self.attention = nn.Sequential(

            nn.Linear(
                lstm_hidden * 2,
                64
            ),

            nn.Tanh(),

            nn.Linear(
                64,
                1
            )
        )

        # =================================================
        # Final Classifier
        # =================================================
        self.classifier = nn.Sequential(

            nn.Linear(
                lstm_hidden * 2,
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
    # 自动推导 LSTM shape
    # =====================================================
    def _get_lstm_shape(self, chans, time_point):

        with torch.no_grad():

            x = torch.zeros(
                1,
                1,
                chans,
                time_point
            )

            x = self.firstconv(x)

            x = self.depthwiseConv(x)

            x = self.separableConv(x)

            # x shape:
            # [B, C, 1, T]

            _, c, h, w = x.shape

            # sequence length = w
            # feature dim = c*h

            feature_dim = c * h
            sequence_length = w

        return feature_dim, sequence_length

    # =====================================================
    # Attention Pooling
    # =====================================================
    def attention_pooling(self, lstm_output):

        # lstm_output:
        # [B, T, 2H]

        attn_weights = self.attention(
            lstm_output
        )

        # [B, T, 1]

        attn_weights = torch.softmax(
            attn_weights,
            dim=1
        )

        weighted_output = lstm_output * attn_weights

        pooled = weighted_output.sum(dim=1)

        return pooled

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, x):

        # 输入:
        # [B, C, T]

        x = x.unsqueeze(1)

        # -> [B, 1, C, T]

        # =================================================
        # EEGNet Feature Extraction
        # =================================================
        x = self.firstconv(x)

        x = self.depthwiseConv(x)

        x = self.separableConv(x)

        # =================================================
        # Prepare for LSTM
        # =================================================
        # x shape:
        # [B, C, 1, T]

        b, c, h, w = x.shape

        # -> [B, T, C*H]
        x = x.permute(0, 3, 1, 2)

        x = x.reshape(
            b,
            w,
            c * h
        )

        # =================================================
        # BiLSTM
        # =================================================
        lstm_output, _ = self.bilstm(x)

        # shape:
        # [B, T, 2H]

        # =================================================
        # Attention Pooling
        # =================================================
        x = self.attention_pooling(
            lstm_output
        )

        # =================================================
        # Classification
        # =================================================
        x = self.classifier(x)

        return x