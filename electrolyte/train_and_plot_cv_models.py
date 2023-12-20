import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from slimnet.chemprop.args import TrainArgs, PredictArgs
from slimnet.chemprop.train import cross_validate, run_training, make_predictions
from chemproppred.utils import plot_hexbin
from chemproppred.make_balanced_train import make_balanced_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib.offsetbox import AnchoredText


PATH_CHEM = os.getcwd()
DATADIR = f"{PATH_CHEM}/data/cross_val_data"
TYPE = "arr"
MODELDIR = f"{PATH_CHEM}/models"

def make_training_predictions(data_path, model_path, gpu=False):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for ROUNDNUM in range(1,6):
        print(ROUNDNUM)
        for TESTNUM in range(2,12):  

            TRAIN=f"{data_path}/s_train_{TESTNUM}.csv"
            TRAINFEATS=f"{data_path}/f_train_{TESTNUM}.csv"

            VAL=f"{data_path}/s_cv_{TESTNUM}.csv"
            VALFEATS=f"{data_path}/f_cv_{TESTNUM}.csv"

            TEST=f"{data_path}/s_test_{TESTNUM}.csv"
            TESTFEATS=f"{data_path}/f_test_{TESTNUM}.csv"

            PREDS=f"{data_path}/{TYPE}/preds_{TESTNUM}_{ROUNDNUM}.csv"
            SAVEDIR=f"{model_path}/checkpoints/check{ROUNDNUM}_{TESTNUM}"

            argument = [
                "--data_path",f"{TRAIN}",
                "--features_path", f"{TRAINFEATS}",
                "--separate_val_path", f"{VAL}",
                "--separate_val_features_path", f"{VALFEATS}",
                "--separate_test_path", f"{TEST}",
                "--separate_test_features_path", f"{TESTFEATS}",
                "--save_dir", f"{SAVEDIR}",
                "--dataset_type", "regression",
                "--metric", "mae",
                "--outputmode", 'slimnet', #"arr",
                "--quiet",
                "--depth", "3",
                "--dropout", "0.15",
                "--ffn_num_layers", "3",
                "--hidden_size", "2300",
                "--batch_size", "100",
                "--pytorch_seed", "3",
                "--epochs", "12",
                "--save_smiles_splits"
            ]

            
            # argument.append("--gpu")
            # argument.append("0")
            # else:
            argument.append("--no_cuda")

            train_args = TrainArgs().parse_args(argument)
            
            # TRAIN THE MODEL
            cross_validate(args=train_args, train_func=run_training)
            
            TRAIN_FULL=f"{data_path}/s_full.csv"
            TRAINFEATS_FULL=f"{data_path}/f_full.csv"
            PREDS=f"{data_path}/preds/preds_screen_{ROUNDNUM}_{TESTNUM}.csv"

            pred_args = [
                "--test_path", f"{TRAIN_FULL}",
                "--features_path", f"{TRAINFEATS_FULL}",
                "--checkpoint_dir", f"{SAVEDIR}",
                "--outputmode", "slimnet", #"arr",
                "--preds_path", f"{PREDS}",
            ]

            make_predictions(args=PredictArgs().parse_args(pred_args))

def plot_parity(data_path):
    preds_path = f'{data_path}/preds'
    paths_df = os.listdir(preds_path)
    # print(paths_df[0])

    df_ref = pd.read_csv(f"{preds_path}/{paths_df[9]}"
                         # , encoding='gb2312'
                         )
    conductivities = [df_ref.conductivity.values]
    smiles = df_ref.smiles.values

    for i in paths_df[1:]:
        df = pd.read_csv(f"{preds_path}/{i}")
        print(f"{preds_path}/{i}")
        if (df.smiles.values == smiles).all():
            conductivities.append(df.conductivity.values)
        else:
            raise ValueError(f"The smiles of {paths_df[0]} doesn't line up with {paths_df[1]}")
    print(np.array(conductivities).shape)
    pred_cond = np.array(conductivities).mean(axis=0)
    print(pred_cond.shape)

    TRAIN=f"{data_path}/s_full.csv"
    df_true = pd.read_csv(TRAIN)

    y_true = df_true.conductivity.values
    y_pred = pred_cond

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(mae, rmse, r2)


    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax, hb = plot_hexbin(df_true.conductivity.values,pred_cond, ax, 'linear')
    ax.set_xlabel('Target Ionic Conductivity (S/cm)', fontdict={'size':20})
    ax.set_ylabel('Predicted Ionic Conductivity (S/cm)', fontdict={'size':20})
    ax.set_title('SoftNET conductivity parity plot',fontdict={'size':26})
    at = AnchoredText(f"MAE = {mae:.3f}\nRMSE = {rmse:.3f}\nR^2 = {r2:.3f}", prop=dict(size=10), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Number of points',fontdict={'size':18})
    plt.tick_params(axis='both', which='major', labelsize=16)
    cb.ax.tick_params(labelsize=16)
    plt.savefig(f'{preds_path}/conductivity_parity_plot.png')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing input parameters for cross validation training')
    parser.add_argument('--make_data', choices=['true', 'false'], default='true',
                        help='Determines whether the data should be generated or not')
    parser.add_argument('--train_predict', choices=['true', 'false'], default='true',
                        help='Should the models be trained or not (takes couple of hours)')
    parser.add_argument('--plot_parity', choices=['true', 'false'], default='true',
                        help='Should the data be plotted, works only when data is made and predicted')
    parser.add_argument('--gpu', choices=['true', 'false'], default='false',
                        help='The model is trained on cuda enabled GPU, default false - training on CPU')
    args = parser.parse_args()
    
    if args.make_data == "true":
        print("Creating the cross validation data files for training!")
        make_balanced_data(DATADIR, f'{PATH_CHEM}/data/clean_train_data.csv') 
    if args.train_predict == "true":
        print("Training loop begins!")
        make_training_predictions(DATADIR, MODELDIR, args.gpu==False)
    if args.plot_parity == "true":
        print("Plotting results")
        plot_parity(DATADIR)
        
    
    