import torch
import torch.nn as nn

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=8):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=channel),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = (
            self.shared_MLP(self.avg_pool(x).view(x.size(0), -1))
            .unsqueeze(2)
            .unsqueeze(3)
        )
        maxout = (
            self.shared_MLP(self.max_pool(x).view(x.size(0), -1))
            .unsqueeze(2)
            .unsqueeze(3)
        )
        return self.sigmoid(avgout + maxout)


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ResBlock_CBAM(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=in_places,
                out_channels=places,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=places,
                out_channels=places,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=places,
                out_channels=places * self.expansion,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(places * self.expansion),
        )
        self.cbam = CBAM(channel=places * self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_places,
                    out_channels=places * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(places * self.expansion),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = torch.mean(out, dim=1, keepdim=True)
        return out

