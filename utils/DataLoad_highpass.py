import config
from glob import glob
import torch
from cv2 import COLOR_BGR2YUV, cvtColor
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from tqdm import tqdm

class MyData(Dataset):
    def __init__(self, data_dir, operation):
        self.operation = operation
        self.root_dir = data_dir
        self.images = []
        for ext in ["jpg", "png"]:
            for path in glob(os.path.join(data_dir, "*.%s" % ext)):
                im = cv2.imread(path)
                self.images.append((path, im))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _, image = self.images[index]
        if self.operation == 'train':
            image = cv2.resize(image, (512, 512))
        if self.operation == 'eval':
            image = cv2.resize(image, (768, 768))
        image = cvtColor(image, COLOR_BGR2YUV)
        image = torch.FloatTensor(image / 127.5 - 1.0).to(config.device)
        return image


def load_train(data_dir, batch_size, operation):
    return DataLoader(
        MyData("%s/train" % data_dir, operation=operation),
        shuffle=True,
        batch_size=batch_size,
        drop_last=True
    )

def load_eval(data_dir, batch_size, operation):
    return DataLoader(
        MyData("%s/v" % data_dir, operation=operation),
        shuffle=False,
        batch_size=batch_size,
        drop_last=True
    )

def load_test(data_dir, operation):
    return DataLoader(
        MyData("%s" % data_dir, operation=operation),
        shuffle=False,
        batch_size=1,
        drop_last=True
    )


if __name__ == '__main__':
    val = load_eval("../dataset", operation='eval')
    iterator = tqdm(val)
    for im in iterator:
        print(im.shape)
