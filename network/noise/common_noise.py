# import torch
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# from random import random, randint
# import random as ra
# import kornia
# import math
# from kornia.geometry.transform.imgwarp import get_perspective_transform
# from kornia.geometry.transform.imgwarp import warp_perspective
# from utils import Jpeg_compression
# from config import training_config as cfg


# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()

#     def forward(self, input, type=""):
#         return input[0].clamp(-1, 1) - input[0]

# def jpeg_compression_train(input, type=""):
#     forward_watermarked_images = input[1]
#     jp = Jpeg_compression.JpegCompression(cfg.device)
#     # JpegCompression 会原地调整输入，因此传入副本，避免污染后续攻击。
#     jpeg_image = jp(forward_watermarked_images.clone())
#     # 当前水印载体是 RGB 图像的 B 通道（索引 2）。
#     return jpeg_image[:, [2], :, :].clamp(-1, 1) - input[0]

# class Resize(nn.Module):
#     def __init__(self, down_scale=0.5):
#         super(Resize, self).__init__()
#         self.down_scale = down_scale

#     def forward(self, input, type=""):
#         forward_u_embedded = input[0]
#         noised_down = F.interpolate(
#             forward_u_embedded,
#             size=(
#                 int(self.down_scale * forward_u_embedded.shape[2]),
#                 int(self.down_scale * forward_u_embedded.shape[3]),
#             ),
#             mode="nearest",
#         )
#         noised_up = F.interpolate(
#             noised_down, size=(forward_u_embedded.shape[2], forward_u_embedded.shape[3]), mode="nearest"
#         )
#         return noised_up.clamp(-1,1) - forward_u_embedded

# class MedianBlur(nn.Module):
#     def __init__(self, kernel_size=(3, 3)):
#         super(MedianBlur, self).__init__()
#         self.transform = kornia.filters.MedianBlur(kernel_size=kernel_size)

#     def forward(self, input, type=""):
#         forward_u_embedded = input[0]
#         return self.transform(forward_u_embedded).clamp(-1,1) - forward_u_embedded

# class GaussianNoise(nn.Module):
#     def __init__(self, mean=0, std=0.01, p=1):
#         super(GaussianNoise, self).__init__()
#         self.transform = kornia.augmentation.RandomGaussianNoise(mean=mean, std=std, p=p)

#     def forward(self, input, type=""):
#         image = input[0]
#         return self.transform(image).clamp(-1, 1) - image


# class GaussianBlur(nn.Module):
#     def __init__(self, kernel_size=(3,3), sigma=(2,2), p=1):
#         super(GaussianBlur, self).__init__()
#         self.transform = kornia.augmentation.RandomGaussianBlur(kernel_size=kernel_size, sigma=sigma, p=p)

#     def forward(self, input, type=""):
#         image = input[0]
#         return self.transform(image).clamp(-1, 1) - image

# class Dropout(nn.Module):
#     def __init__(self, prob=0.3):
#         super(Dropout, self).__init__()
#         self.prob = prob
    
#     def forward(self, input, type=""):
#         forward_u_embedded, forward_cover_images = input[0], input[2]
#         mask = torch.Tensor(np.random.choice([0.0, 1.0], forward_u_embedded.shape[2:], p=[self.prob, 1 - self.prob])).to(forward_u_embedded.device)
#         mask = mask.expand_as(forward_u_embedded)
#         output = forward_u_embedded * mask + forward_cover_images[:, [2], :, :] * (1 - mask)
#         return output.clamp(-1,1) - forward_u_embedded

# class SaltPepper(nn.Module):
# 	def __init__(self, prob=0.05):
# 		super(SaltPepper, self).__init__()
# 		self.prob = prob

# 	def sp_noise(self, image, prob):
# 		mask = torch.Tensor(np.random.choice((0, 1, 2), image.shape[2:], p=[1 - prob, prob / 2., prob / 2.])).to(image.device)
# 		mask = mask.expand_as(image)
# 		# 不原地修改 input[0]，否则噪声结果会和自身相减，并污染后续攻击。
# 		noised = torch.where(mask == 1, torch.ones_like(image), image)
# 		noised = torch.where(mask == 2, -torch.ones_like(image), noised)
# 		return noised

# 	def forward(self, input, type=""):
# 		forward_u_embedded = input[0]
# 		return self.sp_noise(forward_u_embedded, self.prob).clamp(-1,1) - forward_u_embedded



import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint
import random as ra
import kornia
import math
from kornia.geometry.transform.imgwarp import get_perspective_transform
from kornia.geometry.transform.imgwarp import warp_perspective
from utils import Jpeg_compression
from config import training_config as cfg

# input = [forward_b_embedded, forward_watermarked_images, forward_cover_images, forward_mask]

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input, type=""):
        return input[0].clamp(-1, 1) - input[0]

def jpeg_compression_train(input, type=""):
    forward_watermarked_images = input[1]
    jp = Jpeg_compression.JpegCompression(cfg.device)
    rgb_jp = jp(forward_watermarked_images)
    return rgb_jp[:, [2], :, :].clamp(-1,1) - input[0]

class Resize(nn.Module):
    def __init__(self, down_scale=0.5):
        super(Resize, self).__init__()
        self.down_scale = down_scale

    def forward(self, input, type=""):
        forward_u_embedded = input[0]
        noised_down = F.interpolate(
            forward_u_embedded,
            size=(
                int(self.down_scale * forward_u_embedded.shape[2]),
                int(self.down_scale * forward_u_embedded.shape[3]),
            ),
            mode="nearest",
        )
        noised_up = F.interpolate(
            noised_down, size=(forward_u_embedded.shape[2], forward_u_embedded.shape[3]), mode="nearest"
        )
        return noised_up.clamp(-1,1) - forward_u_embedded

class MedianBlur(nn.Module):
    def __init__(self, kernel_size=(3, 3)):
        super(MedianBlur, self).__init__()
        self.transform = kornia.filters.MedianBlur(kernel_size=kernel_size)

    def forward(self, input, type=""):
        forward_u_embedded = input[0]
        return self.transform(forward_u_embedded).clamp(-1,1) - forward_u_embedded

class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std=0.01, p=1):
        super(GaussianNoise, self).__init__()
        self.transform = kornia.augmentation.RandomGaussianNoise(mean=mean, std=std, p=p)

    def forward(self, input, type=""):
        image = input[0]
        return self.transform(image).clamp(-1, 1) - image

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=(3,3), sigma=(1,1), p=1):
        super(GaussianBlur, self).__init__()
        self.transform = kornia.augmentation.RandomGaussianBlur(kernel_size=kernel_size, sigma=sigma, p=p)

    def forward(self, input, type=""):
        image = input[0]
        return self.transform(image).clamp(-1, 1) - image

class Dropout(nn.Module):
    def __init__(self, prob=0.15):
        super(Dropout, self).__init__()
        self.prob = prob
    
    def forward(self, input, type=""):
        forward_u_embedded, forward_cover_images = input[0], input[2]
        mask = torch.Tensor(np.random.choice([0.0, 1.0], forward_u_embedded.shape[2:], p=[self.prob, 1 - self.prob])).to(forward_u_embedded.device)
        mask = mask.expand_as(forward_u_embedded)
        output = forward_u_embedded * mask + forward_cover_images[:, [2], :, :] * (1 - mask)
        return output.clamp(-1,1) - forward_u_embedded

class SaltPepper(nn.Module):
	def __init__(self, prob=0.02, salt_val=0.3, pepper_val=-0.3):
		super(SaltPepper, self).__init__()
		self.prob = prob
		self.salt_val = salt_val
		self.pepper_val = pepper_val

	def forward(self, input, type=""):
		forward_u_embedded = input[0]
		noise = torch.zeros_like(forward_u_embedded)
		mask = torch.Tensor(np.random.choice((0, 1, 2), forward_u_embedded.shape[2:],
			p=[1 - self.prob, self.prob / 2., self.prob / 2.])).to(forward_u_embedded.device)
		mask = mask.expand_as(forward_u_embedded)
		noise[mask == 1] = self.salt_val    # salt: +salt_val
		noise[mask == 2] = self.pepper_val  # pepper: +pepper_val
		noised = forward_u_embedded + noise
		return noised.clamp(-1, 1) - forward_u_embedded
