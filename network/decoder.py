import torch
import torch.nn as nn
from network.encoder import spatial_features_extractor
from network.encoder import Bottleneck
from network.ResBlock import ResBlock
from network.Attention import ResBlock_CBAM
from network.simAM import Simam_module
from config import training_config as cfg
from network.ConvBlock import ConvBlock
import torch.nn.functional as F
import torch.nn.init as init

class Decoder(nn.Module):
    def __init__(self, type):
        super(Decoder, self).__init__()
        self.type = type
        if self.type == "tracer":
            self.in_channels = 4
        else:
            self.in_channels = 2

        self.feature_extractor = spatial_features_extractor(in_c=self.in_channels)
        self.attention = ResBlock_CBAM(64, 16)
        self.simam = Simam_module()

        self.b1 = Bottleneck(64, 64)
        self.b2 = Bottleneck(128, 64)
        self.b3 = Bottleneck(192, 64)

        self.down = nn.Sequential(
            nn.InstanceNorm2d(256),
            nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
            nn.InstanceNorm2d(256),
            nn.Tanh(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            
            nn.InstanceNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),

            nn.InstanceNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),

            nn.InstanceNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, cfg.wm_channels, kernel_size=3, padding=1),
        )

        self.conv_wm = ConvBlock(cfg.wm_channels, 1, blocks=2)
        self.fc = nn.Linear(cfg.message_length**2, cfg.message_length)

    def forward(self, x):
        fm = self.feature_extractor(x)

        o = self.b1(fm)
        o = o + self.simam(o)
        o = self.b2(o)
        o = o + self.simam(o)
        o = self.b3(o)
        o = o + self.simam(o)

        message = self.down(o)

        message = self.conv_wm(message)
        message = F.interpolate(message, size=(cfg.message_length, cfg.message_length), mode="nearest")
        message = message.squeeze(1).view(message.size(0), -1)
        message = self.fc(message)

        return message
