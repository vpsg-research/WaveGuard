# Author: Aspertaine
# Date: 2022/6/1 16:38

import torch

def get_accuracy(extract_wm, wm):
    total_cor = 0.0
    total_pixel = wm.size(1) * wm.size(2)
    for i in range(wm.size(0)):
        extract_wm[i, :, :] = torch.where(extract_wm[i, :, :] >= 0.5, 1, 0)
        cor = 1.0 - (torch.sqrt((extract_wm[i, :, :] - wm[i, :, :]) ** 2).sum() / total_pixel)
        total_cor += cor
    acc = total_cor / wm.size(0)
    return abs(acc)


def get_accuracy_binary(extract_wm, wm):
    # print(extract_wm)
    # print(wm)
    total_cor = 0.0
    total_pixel = (wm.size(1) - 1) * wm.size(2)
    for i in range(wm.size(0)):
        extract_wm[i, :, :] = torch.where(extract_wm[i, :, :] >= 0.5, 1, 0)
        cor = 1.0 - (torch.sqrt((extract_wm[i, :-2, :] - wm[i, :-2, :]) ** 2).sum() / total_pixel)
        total_cor += cor
    acc = total_cor / wm.size(0)
    return abs(acc)


if __name__ == '__main__':
    # e_wm = torch.randn(10, 32, 32)
    # e_wm_ = torch.where(e_wm > 0, 1, 0)
    # print(e_wm_)
    # # wm = torch.randn(2, 32, 32)
    # wm = e_wm_
    # # print(wm)
    # wm = torch.where(wm >= 0.5, 1, 0)
    # ans = get_accuracy(e_wm_, wm)
    # print(ans)

    e_wm = torch.randint(0, 2, [3, 5])
    # wm = e_wm.clone()
    # wm[:, 99] = 0
    # wm[:, 98] = 0
    # wm[:, 97] = 0
    # wm[:, 96] = 0
    # wm[:, 95] = 0
    # ans = get_accuracy_binary(e_wm, wm)
    # print(ans)
    print(e_wm[:-1, :])
