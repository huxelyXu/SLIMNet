
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import re
from rdkit.Chem import Draw
from rdkit.Chem import MACCSkeys
import numpy as np
import torch
import torch.nn as nn
import torch.utils
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
import os
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from SoftNet import SoftNet, train_SoftNet, GetLoader
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText

import torch

from evaluation import ThreeDEvaluator
from run import run
from torch.utils.data import Subset

from schnet_concat import SchNet

data = torch.load('/mnt/workspace/xuhan/github-repo/chemprop-master/small_dataset_test/soft-schnet/data_polymer_scaled.pt')
train_idx = np.load('/mnt/workspace/xuhan/github-repo/chemprop-master/small_dataset_test/soft-schnet/train_id.npy')
test_idx = np.load('/mnt/workspace/xuhan/github-repo/chemprop-master/small_dataset_test/soft-schnet/test_id.npy')

train_data = Subset(data, train_idx)
valid_data = Subset(data, train_idx)
test_data = Subset(data, test_idx)

device = 'cuda:0'

# Define model, loss, and evaluation
model = SchNet(energy_and_force=False, cutoff=5.0, 
               num_layers=6, hidden_channels=128, 
               out_channels=4, 
               num_filters=128, num_gaussians=50)
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

model_dict = model.state_dict()
pretrained_dict = torch.load('/mnt/workspace/xuhan/github-repo/chemprop-master/small_dataset_test/soft-schnet/qm9/valid_checkpoint.pt')['model_state_dict']

pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
del pretrained_dict["update_u.lin1.weight"]
del pretrained_dict["update_u.lin2.weight"]
del pretrained_dict["update_u.lin2.bias"]

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# Train and evaluate
import logging
import time

if  __name__ == '__main__':
    logger = logging.getLogger()
    time = time.asctime()
    logger.setLevel(logging.INFO)   # 设置打印级别
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(f"debug "+time+".log", encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logging.info('Start print log......')

run3d = run()
run3d.run(device, train_data, valid_data, test_data, model, loss_func, evaluation,
          epochs=2000, batch_size=1, vt_batch_size=1, lr=0.0005,
          lr_decay_factor=0.5, lr_decay_step_size=300, softnet=False, 
          save_dir='./schnet-concat/transfer/'
          )

