# Author: Aspertaine
# Date: 2022/6/11 0:08

import torch
import pytorch_ssim

def psnr(images1, images2, batch_size):
    # [B, C, H, W]
    total_psnr = torch.tensor(0.0)
    for i in range(batch_size):
        mse = torch.mean((images1[i, :, :, :] - images2[i, :, :, :]) ** 2)
        total_psnr += 20 * torch.log10(2 / torch.sqrt(mse))
    return total_psnr / batch_size


def ssim(images1, images2, batch_size):
    total_ssim = torch.tensor(0.0)
    for i in range(batch_size):
        total_ssim += pytorch_ssim.ssim(images1[[i], :, :, :], images2[[i], :, :, :])
    return total_ssim / batch_size


if __name__ == '__main__':
    # img1 = torch.rand(10, 3, 256, 256)
    # img2 = torch.rand(10, 3, 256, 256)
    # print(pytorch_ssim.ssim(img1, img2))
    #
    # img1.permute(1, 2, 3, 0)
    # print(img1.shape)

    f1 = torch.rand(10, 3, 256, 256)
    f2 = f1
    print(psnr(f1, f2, 10))
    print(ssim(f1, f2, 10))
    for i in range(10):
        print(f1[i, :, :, :].shape)



