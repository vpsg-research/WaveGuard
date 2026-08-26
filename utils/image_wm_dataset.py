import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

def get_transform(img_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

class ImageWatermarkAttrDataset(Dataset):
    def __init__(self, img_dir, wm_dir, img_size=128, attr_path="celebahq"):
        self.img_dir = img_dir
        self.wm_dir = wm_dir
        self.img_size = img_size
        self.transform = get_transform(img_size)
        
        if attr_path.startswith("celebahq"):
            self.attr_path = "/root/autodl-tmp/code_bk/network/noise/stargan/CelebAMask-HQ-attribute-anno.txt"
        else:
            self.attr_path = "/root/autodl-tmp/code_bk/network/noise/stargan/list_attr_celeba.txt"
        
        self.selected_attrs = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
        self.attr2idx = {}
        self.idx2attr = {}
        self.attr_dict = self._preprocess_attrs()
        
        # 构建图片-水印-标签三元组
        self.pairs = self._build_pairs()

        if len(self.pairs) == 0:
            raise ValueError("No matching (image, watermark, attribute) pairs found!")

    def _preprocess_attrs(self):
        """
        预处理属性文件，返回{图片名: 标签}字典
        """
        lines = [line.rstrip() for line in open(self.attr_path, "r")]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        attr_dict = {}
        for line in lines[2:]:
            split = line.split()
            basename = os.path.basename(split[0])
            filename = os.path.splitext(basename)[0]
            values = split[1:]
            
            label = [values[self.attr2idx[attr]] == "1" for attr in self.selected_attrs]
            attr_dict[filename] = label
        return attr_dict

    def _build_pairs(self):
        """
        构建图片、水印、标签的匹配对
        """
        img_names = [f for f in os.listdir(self.img_dir) 
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        
        pairs = []
        for img_name in img_names:
            wm_name = os.path.splitext(img_name)[0] + ".npy"
            wm_path = os.path.join(self.wm_dir, wm_name)
            if not os.path.exists(wm_path):
                print(f"[WARN] Missing wm for {img_name}, skip.")
                continue
            
            img_basename = os.path.splitext(img_name)[0]
            if img_basename not in self.attr_dict:
                print(f"[WARN] Missing attribute for {img_name}, skip.")
                continue
            
            pairs.append((img_name, wm_name, img_basename))
        return pairs

    def __getitem__(self, idx):
        img_name, wm_name, img_basename = self.pairs[idx]
        
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = self.transform(img)
        
        wm = np.load(os.path.join(self.wm_dir, wm_name)).astype(np.float32)
        wm = torch.from_numpy(wm).float().squeeze()
        
        label = self.attr_dict[img_basename]
        label_tensor = torch.FloatTensor(label)
        
        return img, wm, label_tensor

    def __len__(self):
        return len(self.pairs)


def get_loader(img_dir, wm_dir, img_size, batch_size, shuffle=True):
    # dataset = ImageWatermarkDataset(img_dir, wm_dir, img_size)
    dataset = ImageWatermarkAttrDataset(img_dir, wm_dir, img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
