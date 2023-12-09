
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
from torch_geometric.data import Data
import torch
from schnet_concat import SchNet
from evaluation import ThreeDEvaluator
from run import run
from torch.utils.data import Subset
import numpy as np


# import dgl

def plot_parity(y_true, y_pred, name, dpi_num, y_pred_unc=None):
    axmin = min(min(y_true), min(y_pred)) - 0.1 * (max(y_true) - min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.1 * (max(y_true) - min(y_true))
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    sns.set()
    plt.plot([axmin, axmax], [axmin, axmax], '--k')

    plt.errorbar(y_true, y_pred, yerr=y_pred_unc, linewidth=0, marker='o', markeredgecolor='w', c='cornflowerblue',
                 alpha=1, elinewidth=1)
    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))
    ax = plt.gca()
    ax.set_aspect('equal')

    at = AnchoredText(
        f"MAE = {mae:.3f}\nRMSE = {rmse:.3f}\nR^2 = {r2:.3f}", prop=dict(size=10), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    plt.xlabel('Ground-Truth')
    plt.ylabel('Predicted')
    plt.savefig(f'{name}.png', dpi=dpi_num)

    plt.show()
    return


def plot_comp(y_t, y_p, name, dpi_num, num=4, y_pred_unc=None):
    plt.figure(figsize=(13, 13))
    sns.set()
    for i in range(num):
        plt.subplot(2, 2, i + 1)
        y_true = y_t[:, i]
        y_pred = y_p[:, i]
        axmin = min(min(y_true), min(y_pred)) - 0.1 * (max(y_true) - min(y_true))
        axmax = max(max(y_true), max(y_pred)) + 0.1 * (max(y_true) - min(y_true))
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        plt.plot([axmin, axmax], [axmin, axmax], '--k')
        plt.errorbar(y_true, y_pred, yerr=y_pred_unc, linewidth=0, marker='o', markeredgecolor='w', c='cornflowerblue',
                     alpha=1, elinewidth=1)
        plt.xlim((axmin, axmax))
        plt.ylim((axmin, axmax))
        ax = plt.gca()
        ax.set_aspect('equal')
        at = AnchoredText(
            f"MAE = {mae:.3f}\nRMSE = {rmse:.3f}\nR^2 = {r2:.3f}", prop=dict(size=10), frameon=True, loc='upper left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        plt.xlabel(f'Ground-Truth of {i}')
        plt.ylabel(f'Predicted {i}')

    plt.savefig(f'{name}.png', dpi=dpi_num)
    plt.show()





# Load the dataset and split
data = torch.load('data_polymer_scaled.pt')
train_idx = np.load('train_id.npy') #[:50] #846
test_idx = np.load('test_id.npy')
print(test_idx.shape)

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
          save_dir='./schnet-concat50'
          )
