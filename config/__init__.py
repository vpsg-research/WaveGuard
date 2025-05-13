import yaml
from easydict import EasyDict

with open("config/train.yaml", "r") as f:
    training_config = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

with open("config/vis.yaml", "r") as f:
    vis_config = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))