import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random as ra
import kornia
import math
from random import random, randint
from kornia.geometry.transform.imgwarp import get_perspective_transform
from kornia.geometry.transform.imgwarp import warp_perspective
from utils import Jpeg_compression
from config import training_config as cfg
from network.noise.simswap.test_one_image_vis import SimSwap
from network.noise.ganimation.main_for_vis import GANimation
from network.noise.stargan.main_vis import StarGAN
from network.noise.uniface.swap import UniFaceSwap
# from network.noise.uniface.reenact import UniFaceReenactment
from network.noise.fsrt.new_test import Fsrt
from network.noise.cscs.test import CSCS
from network.noise.stylemask.test import StyleMaskModel
from network.noise.hififace.test import HifiFaceNoise
# from network.noise.DiffSwap.diffswap_noise import DiffSwapNoiseLayer
from network.noise.infoswap.test import InfoSwapNoise
from network.noise.RAFSwap.test import RAFSwapNoise


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class Resize(nn.Module):
    def __init__(self, down_scale=0.5):
        super(Resize, self).__init__()
        self.down_scale = down_scale

    def forward(self, input):
        noised_down = F.interpolate(
            input,
            size=(int(self.down_scale * input.shape[2]), int(self.down_scale * input.shape[3])),
            mode="nearest"
        )
        noised_up = F.interpolate(noised_down, size=(input.shape[2], input.shape[3]), mode="nearest")
        return noised_up

class MedianBlur(nn.Module):
    def __init__(self, kernel_size=(3, 3)):
        super(MedianBlur, self).__init__()
        self.transform = kornia.filters.MedianBlur(kernel_size=kernel_size)

    def forward(self, input):
        return self.transform(input)

class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std=10, p=1):
        super(GaussianNoise, self).__init__()
        self.transform = kornia.augmentation.RandomGaussianNoise(mean=mean, std=std, p=p)

    def forward(self, input):
        return self.transform(input)

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=(3,3), sigma=(2,2), p=1):
        super(GaussianBlur, self).__init__()
        self.transform = kornia.augmentation.RandomGaussianBlur(kernel_size=kernel_size, sigma=sigma, p=p)

    def forward(self, input):
        return self.transform(input)

class Dropout(nn.Module):
    def __init__(self, prob=0.3):
        super(Dropout, self).__init__()
        self.prob = prob
    
    def forward(self, input):
        forward_images_embedded, forward_cover_images = input
        mask = torch.Tensor(np.random.choice([0.0, 1.0], forward_images_embedded.shape[2:], p=[self.prob, 1 - self.prob])).to(forward_images_embedded.device)
        mask = mask.expand_as(forward_images_embedded)
        output = forward_images_embedded * mask + forward_cover_images * (1.0 - mask)
        return output

class SaltPepper(nn.Module):
    def __init__(self, prob=0.05):
        super(SaltPepper, self).__init__()
        self.prob = prob
        
    def sp_noise(self, image, prob):
        mask = torch.Tensor(np.random.choice((0, 1, 2), image.shape[:2], p=[1 - prob, prob / 2., prob / 2.]))
        mask = mask.unsqueeze(-1).expand_as(image)
        image[mask == 1] = 255.0
        image[mask == 2] = 0.0
        return image
    
    def forward(self, input):
        input = torch.FloatTensor(input)
        return np.clip(self.sp_noise(input, self.prob).numpy(), 0, 255).astype(np.uint8)
        

def jpeg_compression_train(input):
    jp = Jpeg_compression.JpegCompression(cfg.device)
    return jp(input)

def stargan_male(input, save_path):
    sg = StarGAN()  # 固定 Male
    return sg(input, save_path, attr='Male')

def ganimation(input, save_path):
    gi = GANimation()
    return gi(input, save_path)

def simswap(input, save_path):
    ss = SimSwap()
    return ss(input, save_path)


def stargan_black_hair(input, save_path):
    sg = StarGAN()
    return sg(input, save_path, attr='Black_Hair')

def stargan_blond_hair(input, save_path):
    sg = StarGAN()
    return sg(input, save_path, attr='Blond_Hair')

def stargan_brown_hair(input, save_path):
    sg = StarGAN()
    return sg(input, save_path, attr='Brown_Hair')

def stargan_young(input, save_path):
    sg = StarGAN()
    return sg(input, save_path, attr='Young')


def uniface(input, save_path):
    uni = UniFaceSwap()
    return uni(input)  # ✅ 实际上没有用 save_path

# def uniface_reenact(input, save_path):
#     uni_ree = UniFaceReenactment()
#     return uni_ree(input)  # ✅ 实际上没有用 save_path

def fsrt(input, save_path):
    fs = Fsrt()
    return fs(input)  


def cscs(input,save_path):
    cs = CSCS()
    return cs(input)



def stylemask(input,save_path):
    SM = StyleMaskModel()
    return SM(input)

def hififace(input,save_path):
    hi = HifiFaceNoise()
    return hi(input)

def infoswap(input,save_path):
    di = InfoSwapNoise()
    return di(input)

def RAFSwap(input,save_path):
    RA  = RAFSwapNoise()
    return RA(input)