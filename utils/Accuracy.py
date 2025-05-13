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
    total_cor = 0.0
    total_pixel = (wm.size(1) - 1) * wm.size(2)
    for i in range(wm.size(0)):
        extract_wm[i, :, :] = torch.where(extract_wm[i, :, :] >= 0.5, 1, 0)
        cor = 1.0 - (torch.sqrt((extract_wm[i, :-2, :] - wm[i, :-2, :]) ** 2).sum() / total_pixel)
        total_cor += cor
    acc = total_cor / wm.size(0)
    return abs(acc)
