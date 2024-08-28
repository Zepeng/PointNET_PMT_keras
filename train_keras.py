#import torch
#from torch.utils.data import DataLoader
#from torch.utils.data import TensorDataset
#from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
# import wandb
import sys
import os
import argparse
from tqdm import tqdm
from pprint import pprint
from time import time
import tensorflow as tf
## .py imports ##
# from PointNet_keras import *
from PointNet_merge import * # change this line for the one above for just pure keras
from read_point_cloud import * 
from utils import *
import datetime
from sklearn.preprocessing import MinMaxScaler

# tensorboard
log_dir = "/home/amigala/tflogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_writer = tf.summary.create_file_writer(log_dir)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

strt = time()
parser = argparse.ArgumentParser()
## Hyperparameters
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--use_wandb', action="store_true")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--smaller_run', action="store_true")
parser.add_argument('--dim_reduce_factor', type=float, default=2)
parser.add_argument('--debug', action="store_true")
parser.add_argument('--save_ver', default=0)
parser.add_argument('--seed', type=int, default=999)
parser.add_argument('--enc_a', type=int, default=8)
parser.add_argument('--enc_b', type=int, default=0)
parser.add_argument('--dec_a', type=int, default=8)
parser.add_argument('--dec_b', type=int, default=0)
parser.add_argument('--o_int_bits', type=int, default=0)
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

# check if file exists
if not os.path.exists(ver):
    os.makedirs(ver)

## file save name
save_name = f"./{ver}/pointNET_keras.weights"


## Load/Preprocess Data
## Load data
pmtxyz = get_pmtxyz("/home/amigala/PointNET_PMT_keras/data/pmt_xyz.dat")
#X.to(torch.float32)
#y.to(torch.float32)
data_npz = np.load('/home/amigala/PointNET_PMT_keras/data/train_X_y_ver_all_xyz_energy.npz')
X_tf = tf.convert_to_tensor(data_npz['X'], dtype=tf.float32)
y_tf = tf.convert_to_tensor(data_npz['y'], dtype=tf.float32)
if args.debug:
    small = 5000
    X_tf, y_tf = X_tf[:small], y_tf[:small]

## switch to match Aobo's syntax (time, charge, x, y, z) -> (x, y, z, label, time, charge)
## insert "label" feature to tensor. This feature (0 or 1) is the activation of sensor
new_X = preprocess_features(X_tf)

# do scaling
target_scaler = MinMaxScaler((-1,1))
print(y_tf.shape)
print(y_tf)
y_tf = target_scaler.fit_transform(y_tf)
y_tf = tf.convert_to_tensor(y_tf)

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
                feat_options=QBitOptions(args.enc_a, args.enc_b),
                decoder_options=QBitOptions(args.dec_a, args.dec_b),
                o_int_bits=args.o_int_bits
                )
epochs = range(args.epochs)

# 2. Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

#model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
model.compile(optimizer=optimizer)

# log graph of model
tb_callback = tf.keras.callbacks.TensorBoard(log_dir)
tb_callback.set_model(model)
print(model.summary())

## Train Loop

# wandb.login()
# run = wandb.init(
#     # Set the project where this run will be logged
#     project="pointnet-keras"
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
        
        # Forward pass
        with tf.GradientTape() as tape:
            out = model(X)
            energy_mult = 160  # scale energy loss
            # Scale the last dimension of 'out' and 'y' tensors
            # out = tf.concat([out[:, :-1], energy_mult * tf.expand_dims(out[:, -1], axis=-1)], axis=-1)
            # y = tf.concat([y[:, :-1], energy_mult * tf.expand_dims(y[:, -1], axis=-1)], axis=-1)

            #out[:, -1], y[:, -1] = energy_mult * out[:, -1], energy_mult * y[:, -1]
            loss = tf.reduce_mean(tf.keras.losses.MSE(out, y))
        
        # Calculate gradients and update weights
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Log loss
        total_loss += loss.numpy()
    
    # Calculate average training loss for the epoch
    total_loss /= len(train_loader)
    # wandb.log({"loss":total_loss})
    tot_train_lst.append(total_loss)
    pbar.update(1)
    # with log_writer.as_default():
    #     tf.summary.scalar(
    #         'loss',
    #         tf.keras.metrics.MSE('loss').result(),
    #         step=epoch
    #     )
    
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
            # if args.save_best:
            #     model.save(save_name + ".keras")
            print("new best found..not saving")
    
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
scale_factor = 1#25.
with tqdm(total=len(val_loader), mininterval=5) as pbar:
    total_val_loss = 0

    for i, batch in enumerate(val_loader):
        X, y = batch
        # try:
        #     X = tf.convert_to_tensor(X.numpy().reshape((BATCH_SIZE, 2126, 1, 6)))
        # except ValueError:
        #     print("skipping batch due to incompatible size")
        #     break
        out = model(X)
        # do inverse transform on data
        # new_shape = out.shape
        # print(X.shape)
        # print(out.shape)
        # print(y.shape)
        out = tf.convert_to_tensor(target_scaler.inverse_transform(out))
        y = tf.convert_to_tensor(target_scaler.inverse_transform(y))
        # print(out)
        # print(y)
        # assert 0

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

# plot_reg(diff=diff, dist=dist, total_val_loss=total_val_loss, abs_diff=abs_diff, save_name=save_name, args=args)
# if args.xyz_energy:
abs_x_diff, abs_y_diff, abs_z_diff, abs_energy_diff = tf.reduce_mean(abs_diff, axis=0)
energy_diff = tf.concat(diff["energy"], axis=0).cpu()
energy_pred = tf.concat(dist["energy_pred"], axis=0).cpu()
energy = tf.concat(dist["energy"], axis=0).cpu()
# else:
#     abs_x_diff, abs_y_diff, abs_z_diff = tf.reduce_mean(abs_diff, axis=0)

x_diff = tf.concat(diff["x"], axis=0).cpu()
y_diff = tf.concat(diff["y"], axis=0).cpu()
z_diff = tf.concat(diff["z"], axis=0).cpu()

x_pred = tf.concat(dist["x_pred"], axis=0).cpu()
y_pred = tf.concat(dist["y_pred"], axis=0).cpu()
z_pred = tf.concat(dist["z_pred"], axis=0).cpu()

x = tf.concat(dist["x"], axis=0).cpu()
y = tf.concat(dist["y"], axis=0).cpu()
z = tf.concat(dist["z"], axis=0).cpu()

# create save data
val_save_data = {
    'abs_x_diff': abs_x_diff.numpy(),
    'abs_y_diff': abs_y_diff.numpy(),
    'abs_z_diff': abs_z_diff.numpy(),
    'abs_energy_diff': abs_energy_diff.numpy(),
    'energy_diff': energy_diff.numpy(),
    'energy_pred': energy_pred.numpy(),
    'energy': energy.numpy(),
    'x_diff': x_diff.numpy(),
    'y_diff': y_diff.numpy(),
    'z_diff': z_diff.numpy(),
    'x_pred': x_pred.numpy(),
    'y_pred': y_pred.numpy(),
    'z_pred': z_pred.numpy(),
    'x': x.numpy(),
    'y': y.numpy(),
    'z': z.numpy(),
    'total_val_loss': total_val_loss
}

import pickle
with open('save_data.pickle', 'wb') as handle:
    pickle.dump(val_save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.close()
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 15))
# plt.subplots_adjust(wspace=0.2)
fig.suptitle(f"Val. MSE: {total_val_loss:.2f} (MSE(x) + MSE(y) + MSE(y) + MSE(energy))\n\
Avg. abs. diff. in x={abs_x_diff:.2f}, y={abs_y_diff:.2f}, z={abs_z_diff:.2f}, energy={abs_energy_diff:.2f}", fontsize=20)
# else:
#     fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
#     fig.suptitle(f"Val. MSE: {total_val_loss:.2f} (MSE(x) + MSE(y) + MSE(y))\n\
#     Avg. abs. diff. in x={abs_x_diff:.2f}, y={abs_y_diff:.2f}, z={abs_z_diff:.2f}")

## diff. plots
x_diff_range = (-50, 50)
large_fontsize = 20
axes[0,0].hist(x_diff, bins=20, range=x_diff_range, edgecolor='black')
axes[0,0].set_title(r"x_diff ($x - \hat{x}$)", fontsize=large_fontsize)
axes[0,0].set_xlabel('x diff', fontsize=large_fontsize)
axes[0,0].set_ylabel('freq', fontsize=large_fontsize)

y_diff_range = (-50, 50)
axes[0,1].hist(y_diff, bins=20, range=y_diff_range, edgecolor='black')
axes[0,1].set_title(r"y_diff ($y - \hat{y}$)", fontsize=large_fontsize)
axes[0,1].set_xlabel('y diff', fontsize=large_fontsize)
# axes[0,1].set_ylabel('freq', fontsize=large_fontsize)

z_diff_range = (-50, 50)
axes[0,2].hist(z_diff, bins=20, range=z_diff_range, edgecolor='black')
axes[0,2].set_title(r"z_diff ($z - \hat{z}$)", fontsize=large_fontsize)
axes[0,2].set_xlabel('z diff', fontsize=large_fontsize)
# axes[0,2].set_ylabel('freq', fontsize=large_fontsize)

energy_diff_range = (0, 1)
axes[0,3].hist(energy_diff, bins=20, range=energy_diff_range, edgecolor='black')
axes[0,3].set_title(r"energy_diff ($energy - \hat{energy}$)", fontsize=large_fontsize)
axes[0,3].set_xlabel('energy diff', fontsize=large_fontsize)
# axes[0,3].set_ylabel('freq', fontsize=large_fontsize)

## dist. plots
x_range = (-250, 250)
axes[1,0].hist(x, bins=20, range=x_range, edgecolor='black', label="x")
axes[1,0].hist(x_pred, bins=20, range=x_range, edgecolor='blue', label=r'$\hat{x}$', alpha=0.5)
axes[1,0].set_title("x dist", fontsize=large_fontsize)
axes[1,0].set_xlabel('x', fontsize=large_fontsize)
axes[1,0].set_ylabel('freq', fontsize=large_fontsize)

y_range = (-250, 250)
axes[1,1].hist(y, bins=20, range=y_range, edgecolor='black', label="y")
axes[1,1].hist(y_pred, bins=20, range=y_range, edgecolor='blue', label=r'$\hat{y}$', alpha=0.5)
axes[1,1].set_title("y dist", fontsize=large_fontsize)
axes[1,1].set_xlabel('y (cm)', fontsize=large_fontsize)
# axes[1,1].set_ylabel('freq', fontsize=large_fontsize)

z_range = (-250, 250)
axes[1,2].hist(x, bins=20, range=x_range, edgecolor='black', label="z")
axes[1,2].hist(x_pred, bins=20, range=x_range, edgecolor='blue', label=r'$\hat{z}$', alpha=0.5)
axes[1,2].set_title("z dist", fontsize=large_fontsize)
axes[1,2].set_xlabel(r'z', fontsize=large_fontsize)
# axes[1,2].set_ylabel('freq', fontsize=large_fontsize)

energy_range = (0, 4)
axes[1,3].hist(energy, bins=20, range=energy_range, edgecolor='black', label="label")
axes[1,3].hist(energy_pred, bins=20, range=energy_range, edgecolor='blue', label="pred", alpha=0.5)
axes[1,3].set_title(r"energy_diff ($energy - \hat{energy}$)", fontsize=large_fontsize)
axes[1,3].set_xlabel('Energy diff (MeV)', fontsize=large_fontsize)
# axes[1,3].set_ylabel('freq', fontsize=large_fontsize)

axes[1, 0].legend()
axes[1, 1].legend()
axes[1, 2].legend()

plt.savefig(f'./{args.save_ver}/{args.save_ver}_hist.png')
plt.close()
# also upload the saved image to wandb
# wandb.log({"plot_reg": wandb.Image(save_name + "_hist.png")})
