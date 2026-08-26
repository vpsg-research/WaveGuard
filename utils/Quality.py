import torch
import pytorch_ssim


def psnr(images1, images2):
    """计算 PSNR，输入范围 [-1, 1]。"""
    batch_size = images1.size(0)
    total_psnr = torch.tensor(0.0, device=images1.device)
    for i in range(batch_size):
        mse = torch.mean((images1[i] - images2[i]) ** 2)
        total_psnr += 20 * torch.log10(2 / torch.sqrt(mse + 1e-8))
    return total_psnr / batch_size


def ssim(images1, images2):
    """计算 SSIM，输入范围 [-1, 1]，内部转换为 [0, 1] 以匹配 pytorch_ssim 常数。"""
    batch_size = images1.size(0)
    total_ssim = torch.tensor(0.0, device=images1.device)
    for i in range(batch_size):
        img1_01 = (images1[i:i+1] * 0.5 + 0.5).clamp(0, 1)
        img2_01 = (images2[i:i+1] * 0.5 + 0.5).clamp(0, 1)
        total_ssim += pytorch_ssim.ssim(img1_01, img2_01)
    return total_ssim / batch_size
