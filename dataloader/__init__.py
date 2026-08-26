import os
from .dataset import attrsImgDataset
from .dataset_vis import attrsImgDataset_vis
from config import training_config as cfg
from config import vis_config as vis_cfg
from torch.utils.data import DataLoader

train_dataset = attrsImgDataset(
    os.path.join(cfg.dataset_path, "train_" + str(cfg.image_size)),
    cfg.image_size,
    "celebahq",
)

val_dataset = attrsImgDataset(
    os.path.join(cfg.dataset_path, "val_" + str(cfg.image_size) + "_small"),
    cfg.image_size,
    "celebahq",
)

if len(val_dataset) == 0:
    raise RuntimeError(
        "验证集为空，请检查目录和文件命名：{}".format(
            val_dataset.image_dir
        )
    )

# val_dataset = attrsImgDataset(
#     "/root/autodl-tmp/CelebAMask-HQ/test/test_256",
#     cfg.image_size,
#     "celebahq",
# )
# # val_dataset = attrsImgDataset(
# #     path="/root/autodl-tmp/LFW/lfw-256",
# #     image_size=cfg.image_size,
# #     attr_path="/root/autodl-tmp/LFW/CelebAMask-HQ-attribute-anno.txt",  # 显式给定
# # )
# print("[✅DEBUG] 当前 val_dataset 路径：", val_dataset.image_dir)




train_dataloader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
)



vis_dataset = attrsImgDataset_vis(
    vis_cfg.dataset_path,
    vis_cfg.image_size,
    "celebahq",
)

vis_dataloader = DataLoader(
    vis_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)
