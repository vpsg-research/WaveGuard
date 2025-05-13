# Author: Aspertaine
# Date: 2022/7/6 15:51

import torch
import config
from pytorch_wavelets import DTCWTForward, DTCWTInverse


# [B, C, H, W] -> [B, C, D, H, W, R]
def images_U_dtcwt_with_low(images_U):
    xfm = DTCWTForward(J=2, biort='near_sym_b', qshift='qshift_b').to(config.device)
    low_pass, high_pass = xfm(images_U)
    return low_pass, high_pass

def dtcwt_images_U(low_pass, high_pass):
    ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').to(config.device)
    return ifm((low_pass, high_pass))

def images_U_dtcwt_without_low(images_U):
    xfm = DTCWTForward(J=2, biort='near_sym_b', qshift='qshift_b').to(config.device)
    _, high_pass = xfm(images_U)
    return high_pass


if __name__ == '__main__':
    image = torch.randn(1, 1, 256, 256).to(config.device)
    l, h = images_U_dtcwt_with_low(image)
    print(l.shape)
    print(h[0].shape)
