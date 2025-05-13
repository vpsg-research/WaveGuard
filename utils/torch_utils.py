import os
import torch
import random
import numpy as np
from config import training_config as cfg
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from torchvision import transforms

indices_encoder = torch.tensor([0, 2]).to(cfg.device)
indices_decoder = torch.tensor([0, 2, 3, 5]).to(cfg.device)
transfroms = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def yuv2rgb(yuv_image):
    assert yuv_image.shape[1] == 3, "input should be 3 channels (YUV)"

    Y = yuv_image[:, 0:1, :, :]
    U = yuv_image[:, 1:2, :, :]
    V = yuv_image[:, 2:3, :, :]

    U = U - 128.0
    V = V - 128.0

    R = Y + 1.402 * V
    G = Y - 0.344136 * U - 0.714136 * V
    B = Y + 1.772 * U

    rgb_image = torch.cat([R, G, B], dim=1)

    rgb_image = torch.clamp(rgb_image, 0, 255)

    return rgb_image


def rgb2yuv(rgb_image):
    assert rgb_image.shape[1] == 3, "input should be 3 channels (RGB)"

    R = rgb_image[:, 0:1, :, :]
    G = rgb_image[:, 1:2, :, :]
    B = rgb_image[:, 2:3, :, :]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = (B - Y) * 0.492 + 128
    V = (R - Y) * 0.877 + 128

    yuv_image = torch.cat([Y, U, V], dim=1)

    yuv_image = torch.clamp(yuv_image, 0, 255)

    return yuv_image


def images_U_dtcwt_with_low(images_U):
    xfm = DTCWTForward(J=2, biort="near_sym_b", qshift="qshift_b").to(cfg.device)
    low_pass, high_pass = xfm(images_U)
    return low_pass, high_pass


def dtcwt_images_U(low_pass, high_pass):
    ifm = DTCWTInverse(biort="near_sym_b", qshift="qshift_b").to(cfg.device)
    return ifm((low_pass, high_pass))


def images_U_dtcwt_without_low(images_U):
    xfm = DTCWTForward(J=2, biort="near_sym_b", qshift="qshift_b").to(cfg.device)
    _, high_pass = xfm(images_U)
    return high_pass


def preprocess(images):
    images_Y = images[:, [0], :, :]
    images_U = images[:, [1], :, :]
    images_V = images[:, [2], :, :]
    low_pass, high_pass = images_U_dtcwt_with_low(images_U)
    return images_Y, images_U, images_V, low_pass, high_pass


def normalize(tensor):
    normalizer = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return normalizer(tensor)


def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    # The fmean and std are assumed to be lists with 3 values, one for each channel
    mean = (
        torch.tensor(mean).view(1, 3, 1, 1).to(cfg.device)
    )  # Reshape to match the tensor shape
    std = torch.tensor(std).view(1, 3, 1, 1).to(cfg.device)

    # Denormalize
    tensor = tensor * std + mean
    return tensor


def yuv_normalize(tensor):
    return tensor / 127.5 - 1.0


def yuv_denormalize(tensor):
    return (tensor + 1.0) * 127.5


def decoded_message_error_rate(message, decoded_message):
    length = message.shape[0]

    message = message.gt(0)
    decoded_message = decoded_message.gt(0)
    error_rate = float((message != decoded_message).sum().item()) / length
    return error_rate


def decoded_message_error_rate_batch(messages, decoded_messages):
    error_rate = 0.0
    batch_size = len(messages)
    for i in range(batch_size):
        error_rate += decoded_message_error_rate(messages[i], decoded_messages[i])
    error_rate /= batch_size
    return torch.tensor(error_rate).to(cfg.device)