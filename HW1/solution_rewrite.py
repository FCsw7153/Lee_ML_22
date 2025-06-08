import math
import numpy as np
import pandas as pd
import os
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# 使每次随机数生成器生成的随机数种子都使用相同的种子
def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.ramdom.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 分割数据集为训练集和验证集
def train_valid_split(data_set, valid_ratio, seed):
    valid_set_size = len(data_set) * valid_ratio
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set). np.array(valid_set)

# 数据集类
class COVID19Dataset(Dataset):
    def __init__(self, feature, target):
        if target is None:
            self.target = target
        else:
            self.target = torch.FloatTensor(target)
        self.feature = feature

    def __getitem__(self, idx):
        if self.target is None:
            return self.feature[idx]
        else:
            return self.feature[idx], self.target[idx]
    
    def __len__(self):
        return len(self.feature)