
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
from dig.threedgraph.dataset import QM93D
from dig.threedgraph.dataset import MD17
from dig.threedgraph.method import SphereNet, SchNet, DimeNetPP, ComENet
from dig.threedgraph.method import run
from dig.threedgraph.evaluation import ThreeDEvaluator

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


# import torch
# from schnet import SchNet
# from evaluation import ThreeDEvaluator
# from run import run
from torch.utils.data import Subset
import numpy as np

# Load the dataset and split
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
device = 'cuda:0'
dataset = QM93D(root='dataset/')
target = 'homo' # choose from: mu, alpha, homo, lumo, r2, zpve, U0, U, H, G, Cv
dataset.data.y = dataset.data[target]

split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)

train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))

# Define model, loss, and evaluation
model = SchNet(energy_and_force=False, cutoff=5.0,
               num_layers=6, hidden_channels=128,
               out_channels=1,
               num_filters=128, num_gaussians=50)
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

# Train and evaluate


run3d = run()
run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
          epochs=2500, batch_size=32, vt_batch_size=32, lr=0.005,
          lr_decay_factor=0.7, lr_decay_step_size=300, 
          save_dir='./qm9'
          )
