import torch
#from torch.utils.data import DataLoader
#from torch.utils.data import TensorDataset
#from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
#import matplotlib.pyplot as plt
import wandb
import sys
import os
import argparse
from tqdm import tqdm
from pprint import pprint
from time import time
import tensorflow as tf
## .py imports ##
from PointNet_merge import *
from read_point_cloud import * 
from utils import *

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
parser.add_argument('--seed', type=int, default=999)
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--xyz_label', action="store_true")
parser.add_argument('--xyz_energy', action="store_true")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

args = parser.parse_args()
ver = args.save_ver
#init_logfile(ver)

## file save name
save_name = f"./{ver}/pointNET_qkeras.weights"

scale_factor = 25.
## Load/Preprocess Data
## Load data
pmtxyz = get_pmtxyz("/home/amigala/PointNET_PMT_keras/data/pmt_xyz.dat")
#X.to(torch.float32)
#y.to(torch.float32)
#data_npz = np.load('/expanse/lustre/scratch/zli10/temp_project/pointnet/train_X_y_ver_all_xyz_energy.npz')
#X_tf = tf.convert_to_tensor(data_npz['X']/scale_factor, dtype=tf.float32)
#y_tf = tf.convert_to_tensor(data_npz['y']/scale_factor, dtype=tf.float32)
#y_tf = tf.concat([y_tf[:, :-1], 160* tf.expand_dims(y_tf[:, -1], axis=-1)], axis=-1) #scale the energy by 160
# Load Preprocess data
print("loading data...")
# X, y = torch.load("/expanse/lustre/scratch/zli10/temp_project/pointnet/preprocessed_data.pt")
# X_tf = tf.convert_to_tensor(X.numpy(), dtype=tf.float32)
# y_tf = tf.convert_to_tensor(y.numpy(), dtype=tf.float32)
data_npz = np.load('/home/amigala/PointNET_PMT_keras/data/train_X_y_ver_all_xyz_energy.npz')
X_tf = tf.convert_to_tensor(data_npz['X'], dtype=tf.float32)
y_tf = tf.convert_to_tensor(data_npz['y'], dtype=tf.float32)
if args.debug:
    print("debug got called")
    small = 5000
    X_tf, y_tf = X_tf[:small], y_tf[:small]


# Update batch size
n_data, args.n_hits, F_dim = X_tf.shape

## switch to match Aobo's syntax (time, charge, x, y, z) -> (x, y, z, label, time, charge)
## insert "label" feature to tensor. This feature (0 or 1) is the activation of sensor
new_X = preprocess_features(X_tf)

## Shuffle Data (w/ Seed)
np.random.seed(seed=args.seed)
#set_seed(seed=args.seed)
idx = np.random.permutation(new_X.shape[0]) 
#new_X = tf.gather(new_X, idx)
#y = tf.gather(y_tf, idx)
## Split and Load data
train_split = 0.7
val_split = 0.3
train_idx = int(new_X.shape[0] * train_split)
val_idx = int(train_idx + new_X.shape[0] * train_split)
train = tf.data.Dataset.from_tensor_slices((new_X[:train_idx], y_tf[:train_idx]))
val = tf.data.Dataset.from_tensor_slices((new_X[train_idx:val_idx], y_tf[train_idx:val_idx]))
test = tf.data.Dataset.from_tensor_slices((new_X[val_idx:], y_tf[val_idx:]))
train_loader = train.shuffle(buffer_size=len(new_X)).batch(args.batch_size)
val_loader = val.batch(args.batch_size)
test_loader = val.batch(args.batch_size)
print(f"num. total: {len(new_X)} train: {len(train)}, val: {len(val)}, test: {len(test)}")
print(pmtxyz.shape, tf.shape(new_X), y_tf.shape)
## Init model
model = PointClassifier(
                n_hits=pmtxyz.shape[0], 
                dim = tf.shape(new_X)[2],
                dim_reduce_factor=args.dim_reduce_factor,
                out_dim = y_tf.shape[-1],
                enc_dropout = args.enc_dropout,
                dec_dropout = args.dec_dropout
                )
epochs = range(args.epochs)

# 2. Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, weight_decay=args.weight_decay)

#model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
model.compile(optimizer=optimizer)

## Train Loop
# wandb.login()
# run = wandb.init(
#     # Set the project where this run will be logged
#     project="pointnet-qkeras"
#     # Track hyperparameters and run metadata
#     # config={
#     #     "learning_rate": lr,
#     #     "epochs": epochs,
#     # }
# )

# Initialize tqdm progress bar
pbar = tqdm(total=args.epochs, mininterval=10)

# Initialize best validation loss and best train loss
best_val, best_train = float("inf"), float("inf")
tot_train_lst = []

# Loop through epochs
for epoch in range(args.epochs):
    total_loss = 0
    
    # Loop through batches in training loader
    for i, batch in enumerate(train_loader):
        X, y = batch
        print(X.shape)
        assert 0
        
        # Forward pass
        with tf.GradientTape() as tape:
            out = model(X)
            energy_mult = 160  # scale energy loss
            # Scale the last dimension of 'out' and 'y' tensors
            out = tf.concat([out[:, :-1], energy_mult * tf.expand_dims(out[:, -1], axis=-1)], axis=-1)
            y = tf.concat([y[:, :-1], energy_mult * tf.expand_dims(y[:, -1], axis=-1)], axis=-1)

            #out[:, -1], y[:, -1] = energy_mult * out[:, -1], energy_mult * y[:, -1]
            loss = tf.reduce_mean(tf.keras.losses.MSE(out, y))
        
        # Calculate gradients and update weights
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Log loss
        total_loss += loss.numpy()
    
    # Calculate average training loss for the epoch
    total_loss /= len(train_loader)
    # wandb.log({"Loss":total_loss})
    tot_train_lst.append(total_loss)
    pbar.update(1)
    
    # Adjust learning rate based on training loss
    #prev_lr = optimizer.learning_rate.numpy()
    #reduce_lr.on_epoch_end(epoch, logs={'loss': total_loss})
    #scheduler.step(total_loss)
    
    # Validation every 10 epochs or last epoch
    if epoch == args.epochs - 1:
        total_val_loss = 0
        
        # Validation (evaluation mode)
        for batch in val_loader:
            X, y = batch
            out = model(X)
            val_loss = tf.reduce_mean(tf.keras.losses.MSE(out, y))
            total_val_loss += val_loss.numpy()
        
        # Calculate average validation loss
        total_val_loss /= len(val_loader)
        
        # Update progress bar
        # pbar.update(10)
        
        # Save best model based on validation loss
        if total_val_loss < best_val:
            best_val = total_val_loss
            if args.save_best:
                model.save(save_name + ".keras")
    
    # Log training loss
    else:
        print(total_loss)
    
# Close progress bar
pbar.close()

## time
tot_time = time() - strt

## min values
min_train = round(min(tot_train_lst), 2)
min_values = {
        "min_train": min_train,
        "min_val": best_val,
}

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
        abs_diff.append(tf.abs(y*scale_factor - out*scale_factor))
        val_loss = tf.reduce_mean(tf.keras.losses.MSE(out, y))
        total_val_loss += val_loss.numpy()

        diff_tensor = (y - out)*scale_factor ## to vis. distribution
        dist["x"].append(y[:, 0]*scale_factor)
        dist["y"].append(y[:, 1]*scale_factor)
        dist["z"].append(y[:, 2]*scale_factor)

        dist["x_pred"].append(out[:, 0]*scale_factor)
        dist["y_pred"].append(out[:, 1]*scale_factor)
        dist["z_pred"].append(out[:, 2]*scale_factor)
        
        diff["x"].append(diff_tensor[:, 0])
        diff["y"].append(diff_tensor[:, 1])
        diff["z"].append(diff_tensor[:, 2])

        dist["energy"].append(y[:, 3]*scale_factor)
        dist["energy_pred"].append(out[:, 3]*scale_factor)
        diff["energy"].append(diff_tensor[:, 3])

        pbar.update()
    total_val_loss /= len(val_loader)

abs_diff = tf.concat(abs_diff, axis=0)

## plot and save
plot_reg(diff=diff, dist=dist, total_val_loss=total_val_loss, abs_diff=abs_diff, save_name=save_name, args=args)
