import torch
import numpy as np
import random

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)