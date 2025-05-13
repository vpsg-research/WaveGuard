# Author: Aspertaine
# Date: 2022/7/6 20:09
import torch
import cv2

def watermark_generator():
    random_w = torch.randint(0, 2, [32, 32]).to(torch.float32)
    return random_w.view(1, 1, 32, 32)


def watermark_expend(watermark, batch_size, channels, H, W):
    return watermark.expand(batch_size, channels, H, W)


if __name__ == '__main__':
    """
    output : [1, 1, 32, 32]
    """
    a = watermark_generator().squeeze(0)
    a = a.view(32, 32, 1)
    cv2.imwrite("dd.png", (torch.where(a * 255 == 255, 255, 0)).numpy())
    print(a)
