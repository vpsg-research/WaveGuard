import torch
import torch.nn as nn
import torch.nn.functional as F
from network.Attention import ResBlock_CBAM
from network.ConvBlock import ConvBlock
from network.ResBlock import ResBlock
from network.simAM import Simam_module
from config import training_config as cfg
import torch.nn.init as init

# 空间结构特征提取
class spatial_features_extractor(nn.Module):
    def __init__(self, in_c=2):
        super(spatial_features_extractor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        spatial = self.net(x)
        return spatial

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, 4 * out_channels, kernel_size=1)
        self.bn2 = nn.InstanceNorm2d(4 * out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(4 * out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.attention = ResBlock_CBAM(64, 16)
        self.simam = Simam_module()

        self.linear0 = nn.Linear(cfg.message_length, cfg.message_length**2)
        self.linear1 = nn.Linear(cfg.message_length, cfg.message_length**2)
        self.linear2 = nn.Linear(cfg.message_length, cfg.message_length**2)
        wm_channels = [cfg.wm_channels, cfg.wm_channels, cfg.wm_channels]
        self.conv_wm_0 = ConvBlock(1, wm_channels[0], blocks=2)
        self.conv_wm_1 = ConvBlock(1, wm_channels[1], blocks=2)
        self.conv_wm_2 = ConvBlock(1, wm_channels[2], blocks=2)

        self.feature_extractor = spatial_features_extractor()
        # 多阶段 Bottleneck 残差融合
        self.b0 = Bottleneck(64 + wm_channels[0], 64)
        self.b1 = Bottleneck(64 * 2 + wm_channels[0] + wm_channels[1], 64)
        self.b2 = Bottleneck(64 * 3 + wm_channels[0] + wm_channels[1] + wm_channels[2], 64)

        self.down = nn.Sequential(
            nn.InstanceNorm2d(64 * 3 + wm_channels[0] + wm_channels[1] + wm_channels[2] + 64),
            nn.Tanh(),
            nn.Conv2d (64 * 3 + wm_channels[0] + wm_channels[1] + wm_channels[2] + 64, 256, kernel_size=3, padding=1),

            nn.InstanceNorm2d(256),
            nn.Tanh(),
            nn.Conv2d (256, 128, kernel_size=3, padding=1),

            nn.InstanceNorm2d(128),
            nn.Tanh(),
            nn.Conv2d (128, 64, kernel_size=3, padding=1),
            
            nn.InstanceNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, cfg.wm_channels, kernel_size=3, padding=1),
            
            nn.InstanceNorm2d(cfg.wm_channels),
            nn.Tanh(),
            nn.Conv2d(cfg.wm_channels, 2, kernel_size=3, padding=1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        for linear_layer in [self.linear0, self.linear1, self.linear2]:
            init.xavier_uniform_(linear_layer.weight)
            init.constant_(linear_layer.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.running_mean, 0)
                torch.nn.init.constant_(m.running_var, 1)

    def forward(self, x, watermark):
        fm = self.feature_extractor(x)
        attention_weight = self.attention(fm)

        wm_ex_0 = self.linear0(watermark)
        wm_ex_0 = wm_ex_0.view(-1, 1, cfg.message_length, cfg.message_length)
        wm_ex_0 = F.interpolate(wm_ex_0, size=(fm.shape[2], fm.shape[3]), mode="nearest")
        wm_ex_0 = self.conv_wm_0(wm_ex_0)
        o = self.b0(torch.cat([fm, wm_ex_0], dim=1))
        o = o + self.simam(o)

        wm_ex_1 = self.linear1(watermark)
        wm_ex_1 = wm_ex_1.view(-1, 1, cfg.message_length, cfg.message_length)
        wm_ex_1 = F.interpolate(wm_ex_1, size=(fm.shape[2], fm.shape[3]), mode="nearest")
        wm_ex_1 = self.conv_wm_1(wm_ex_1)
        o = self.b1(torch.cat([o, wm_ex_1], dim=1))
        o = o + self.simam(o)

        wm_ex_2 = self.linear2(watermark)
        wm_ex_2 = wm_ex_2.view(-1, 1, cfg.message_length, cfg.message_length)
        wm_ex_2 = F.interpolate(wm_ex_2, size=(fm.shape[2], fm.shape[3]), mode="nearest")
        wm_ex_2 = self.conv_wm_2(wm_ex_2)
        o = self.b2(torch.cat([o, wm_ex_2], dim=1))
        o = o + self.simam(o)

        wm_diff = self.down(o)
        
        return x + cfg.wm_factor * attention_weight * wm_diff
