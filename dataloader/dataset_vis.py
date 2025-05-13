import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import cv2
from config import vis_config as cfg


class attrsImgDataset_vis(Dataset):
    def __init__(self, path, image_size, attr_path="celeba"):
        super(attrsImgDataset_vis, self).__init__()
        self.image_size = image_size
        self.image_dir = path
        if attr_path[0 : len("celebahq")] != "celebahq":
            self.attr_path = "/root/autodl-tmp/code_bk/network/noise/stargan/list_attr_celeba.txt"
        else:
            self.attr_path = "/root/autodl-tmp/code_bk/network/noise/stargan/CelebAMask-HQ-attribute-anno.txt"

        self.selected_attrs = [
            "Black_Hair",
            "Blond_Hair",
            "Brown_Hair",
            "Male",
            "Young",
        ]
        self.list = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.transform = transforms.Compose(
            [                                                
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.preprocess()
        

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, "r")]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]

        for i, line in enumerate(lines):
            split = line.split()
            basename = os.path.basename(split[0])
            filename = os.path.splitext(basename)[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == "1")

            if os.path.exists(
                os.path.join(self.image_dir, str(filename).zfill(5) + ".png")
            ):
                self.list.append([str(filename).zfill(5) + ".png", label])

    def __getitem__(self, index):

        """Return one image and its corresponding attribute label."""
        filename, label = self.list[index]
        image = cv2.imread(os.path.join(self.image_dir, filename))
        img_ori = image.copy()

        if image is not None:
            try:
                image = cv2.resize(image, (cfg.image_size, cfg.image_size))
            except cv2.error as e:
                height, width = image.shape[:2]
                print(f"Error resizing image {os.path.join(self.image_dir, filename)} (size: {width}x{height}): {e}")
        else:
            print(f"image is none: {os.path.join(self.image_dir, filename)}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image = torch.FloatTensor(image / 127.5 - 1.0)
        if image is not None:
            label_tensor = torch.FloatTensor(label)
        
        return image.permute(2, 0, 1), label_tensor, img_ori


    def __len__(self):
        return len(self.list)
