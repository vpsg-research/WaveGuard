import torch
from config import training_config as cfg
from pytorch_wavelets import DTCWTForward, DTCWTInverse

#	获取完整 DT-CWT 分解（用于嵌入前）
def images_U_dtcwt_with_low(images_U):
    xfm = DTCWTForward(J=2, biort='near_sym_b', qshift='qshift_b').to(cfg.device)
    low_pass, high_pass = xfm(images_U)
    return low_pass, high_pass
#重建图像，用于将水印嵌入后的频域图像转回RGB图像
def dtcwt_images_U(low_pass, high_pass):
    ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').to(cfg.device)
    return ifm((low_pass, high_pass))
#仅提取高频用于嵌入/解码/追踪
def images_U_dtcwt_without_low(images_U):
    xfm = DTCWTForward(J=2, biort='near_sym_b', qshift='qshift_b').to(cfg.device)
    _, high_pass = xfm(images_U)
    return high_pass



