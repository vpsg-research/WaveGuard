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
    """
    将 YUV 图像转换为 RGB 图像 (PyTorch 版本)
    输入:
        yuv_image: 形状为 (B, C, H, W) 的张量，其中 C=3 (YUV 格式)
    输出:
        rgb_image: 形状为 (B, C, H, W) 的张量，其中 C=3 (RGB 格式)
    """
    # 确保输入的张量具有正确的形状
    assert yuv_image.shape[1] == 3, "输入张量应具有 3 个通道 (YUV)"

    # 将 YUV 通道分开
    Y = yuv_image[:, 0:1, :, :]  # (B, 1, H, W)
    U = yuv_image[:, 1:2, :, :]  # (B, 1, H, W)
    V = yuv_image[:, 2:3, :, :]  # (B, 1, H, W)

    # 调整 U 和 V 的偏移量 (OpenCV 使用 128 作为基准)
    U = U - 128.0
    V = V - 128.0

    # 根据 YUV 到 RGB 的转换公式计算 R、G、B
    R = Y + 1.402 * V
    G = Y - 0.344136 * U - 0.714136 * V
    B = Y + 1.772 * U

    # 拼接 R、G、B 通道
    rgb_image = torch.cat([R, G, B], dim=1)

    # 确保 RGB 图像的范围在 [0, 255] 之间
    rgb_image = torch.clamp(rgb_image, 0, 255)

    return rgb_image


def rgb2yuv(rgb_image):
    """
    将 RGB 图像转换为 YUV 图像 (PyTorch 版本)
    输入:
        rgb_image: 形状为 (B, C, H, W) 的张量，其中 C=3 (RGB 格式)
    输出:
        yuv_image: 形状为 (B, C, H, W) 的张量，其中 C=3 (YUV 格式)
    """
    # 确保输入的张量具有正确的形状
    assert rgb_image.shape[1] == 3, "输入张量应具有 3 个通道 (RGB)"

    # 将 RGB 通道分开
    R = rgb_image[:, 0:1, :, :]  # (B, 1, H, W)
    G = rgb_image[:, 1:2, :, :]  # (B, 1, H, W)
    B = rgb_image[:, 2:3, :, :]  # (B, 1, H, W)

    # 根据 RGB 到 YUV 的转换公式计算 Y、U、V
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = (B - Y) * 0.492 + 128
    V = (R - Y) * 0.877 + 128

    # 拼接 Y、U、V 通道
    yuv_image = torch.cat([Y, U, V], dim=1)

    # 确保 YUV 图像的范围在 [0, 255] 之间
    yuv_image = torch.clamp(yuv_image, 0, 255)

    return yuv_image


# [B, C, H, W] -> [B, C, D, H, W, R]
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


def denormalize(tensor, mean, std):
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
    error_rate = float(sum(message != decoded_message)) / length
    return error_rate


def decoded_message_error_rate_batch(messages, decoded_messages):
    error_rate = 0.0
    batch_size = len(messages)
    for i in range(batch_size):
        error_rate += decoded_message_error_rate(messages[i], decoded_messages[i])
    error_rate /= batch_size
    return error_rate
