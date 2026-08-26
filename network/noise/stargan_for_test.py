from .stargan.main_for_test import StarGAN
from utils.torch_utils import normalize, denormalize, yuv_denormalize, yuv_normalize

def stargan_noise(input, type, c_trg=1):
    stargan_noise = StarGAN()
    df_input = [normalize(yuv_denormalize(input[1])), input[3]]
    noised_image = stargan_noise(df_input, c_trg)
    noised_image = yuv_normalize(denormalize(noised_image))[:, [1], :, :]
    gap = noised_image - input[0]
    return gap if type == "deepfake" else noised_image