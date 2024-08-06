import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import sys
import os
import argparse
from tqdm import tqdm
from time import time
import tensorflow as tf
## .py imports ##
from pointnet_keras import *
from read_point_cloud import * 
from utils import *
from keras.models import load_model
import keras

strt = time()
parser = argparse.ArgumentParser()
## Hyperparameters
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--use_wandb', action="store_true")
parser.add_argument('--reduce_lr_wait', type=int, default=20)
parser.add_argument('--enc_dropout', type=float, default=0.2)
parser.add_argument('--dec_dropout', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--smaller_run', action="store_true")
parser.add_argument('--dim_reduce_factor', type=float, default=1.5)
parser.add_argument('--debug', action="store_true")
parser.add_argument('--save_ver', default=0)
parser.add_argument('--mean_only', action="store_true")
parser.add_argument('--save_best', action="store_true")
parser.add_argument('--xyz_label', action="store_true")
parser.add_argument('--xyz_energy', action="store_true")
parser.add_argument('--seed', type=int, default=999)
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--load_ver', type=int, default=13)
parser.add_argument('--debug_vis', action='store_true')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

args = parser.parse_args()
ver = args.load_ver
## file save name
save_name = f"./{ver}/pointNET.weights"


## Load/Preprocess Data
## Load data
pmtxyz = get_pmtxyz("/expanse/lustre/scratch/zli10/temp_project/pointnet/pmt_xyz.dat")
X, y = torch.load(f"/expanse/lustre/scratch/zli10/temp_project/pointnet/train_X_y_ver_all_xyz_energy.pt", map_location=torch.device("cpu"))
X.to(torch.float32)
y.to(torch.float32)
X_tf = tf.convert_to_tensor(X.numpy(), dtype=tf.float32)
y_tf = tf.convert_to_tensor(y.numpy(), dtype=tf.float32)
if args.debug:
    small = 1000
    X_tf, y_tf = X_tf[:small], y_tf[:small]

## switch to match Aobo's syntax (time, charge, x, y, z) -> (x, y, z, label, time, charge)
## insert "label" feature to tensor. This feature (0 or 1) is the activation of sensor
print("preprocess Data")
new_X = preprocess_features(X_tf)

## Shuffle Data (w/ Seed)
np.random.seed(seed=args.seed)
#set_seed(seed=args.seed)
idx = np.random.permutation(new_X.shape[0]) 
new_X = tf.gather(new_X, idx)
y = tf.gather(y_tf, idx)
## Split and Load data
train_split = 0.7
val_split = 0.3
train_idx = int(new_X.shape[0] * train_split)
val_idx = int(train_idx + new_X.shape[0] * train_split)
train = tf.data.Dataset.from_tensor_slices((new_X[:train_idx], y[:train_idx]))
val = tf.data.Dataset.from_tensor_slices((new_X[train_idx:val_idx], y[train_idx:val_idx]))
test = tf.data.Dataset.from_tensor_slices((new_X[val_idx:], y[val_idx:]))
train_loader = train.shuffle(buffer_size=len(new_X)).batch(args.batch_size)
val_loader = val.batch(args.batch_size)
test_loader = val.batch(args.batch_size)
print(f"num. total: {len(new_X)} train: {len(train)}, val: {len(val)}, test: {len(test)}")
print(pmtxyz.shape, tf.shape(new_X), y.shape)
## Init model
model = PointClassifier(
                n_hits=pmtxyz.shape[0], 
                dim = tf.shape(new_X)[2],
                dim_reduce_factor=args.dim_reduce_factor,
                out_dim = y.shape[-1],
                enc_dropout = args.enc_dropout,
                dec_dropout = args.dec_dropout
                )
#nparam = sum([p.numel() for p in model.parameters()])
# 2. Define the optimizer
#optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, weight_decay=args.weight_decay)

#model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
#model.compile(optimizer=optimizer)
#model.load_weights(save_name+".h5", by_name=True)
model = keras.saving.load_model(save_name + '.keras')
epochs = range(args.epochs)
diff = {"x":[], "y":[], "z":[], "radius": [], "unif_r":[], "energy":[]}
dist = {"x":[], "y":[], "z":[], "x_pred":[], "y_pred":[], "z_pred":[], "energy":[], "energy_pred":[],
         "radius": [], "radius_pred": [], "unif_r": [], "unif_r_pred": []}
abs_diff = []
#if args.train_mode:
#    model.train()
#else:
print('Model eval')
with tqdm(total=len(val_loader), mininterval=5) as pbar:
    total_val_loss = 0

    for i, batch in enumerate(val_loader):
        X, y = batch
        out = model(X)
        abs_diff.append(tf.abs(y - out))
        val_loss = tf.reduce_mean(tf.keras.losses.MSE(out, y))
        total_val_loss += val_loss.numpy()

        diff_tensor = y - out ## to vis. distribution
        dist["x"].append(y[:, 0])
        dist["y"].append(y[:, 1])
        dist["z"].append(y[:, 2])

        dist["x_pred"].append(out[:, 0])
        dist["y_pred"].append(out[:, 1])
        dist["z_pred"].append(out[:, 2])
        
        diff["x"].append(diff_tensor[:, 0])
        diff["y"].append(diff_tensor[:, 1])
        diff["z"].append(diff_tensor[:, 2])

        dist["energy"].append(y[:, 3])
        dist["energy_pred"].append(out[:, 3])
        diff["energy"].append(diff_tensor[:, 3])

        pbar.update()
    total_val_loss /= len(val_loader)

abs_diff = tf.concat(abs_diff, axis=0)

## plot and save
plot_reg(diff=diff, dist=dist, total_val_loss=total_val_loss, abs_diff=abs_diff, save_name=save_name, args=args)
