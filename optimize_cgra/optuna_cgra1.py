import os
import pytest
import itertools
import sys
sys.path.append("../../")
sys.path.append("../")
from tensorflow import keras
from keras.layers import Input
from keras.models import Model, save_model
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
from qkeras.utils import load_qmodel
import numpy as np
import pprint
#from read_point_cloud import *
#from preprocess import *
import tensorflow as tf
#tf.keras.utils.set_random_seed(0)
#import wandb
from tqdm import tqdm
from time import time
from PointNet_merge import *
from read_point_cloud import *
from utils import *
import inspect

from sklearn.preprocessing import MinMaxScaler
import matplotlib
import pickle
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

from deepsocflow import *
import json

#Avi added code
import optuna
import utils


(SIM, SIM_PATH) = ('xsim', "F:/Xilinx/Vivado/2022.2/bin/") if os.name=='nt' else ('verilator', '')
np.random.seed(42)
tf.random.set_seed(42)

DEBUG = False
BATCH_SIZE = 64

# Load and preprocess data outside of the objective function
pmtxyz = get_pmtxyz("/home/amehta/PointNET_PMT_keras/data/pmt_xyz.dat")
data_npz = np.load('/home/amehta/PointNET_PMT_keras/data/train_X_y_ver_all_xyz_energy.npz')
X_tf = tf.convert_to_tensor(data_npz['X'], dtype=tf.float32)
y_tf = tf.convert_to_tensor(data_npz['y'], dtype=tf.float32)

# Scale target data
target_scaler = MinMaxScaler((-1, 1))
y_tf = tf.convert_to_tensor(target_scaler.fit_transform(y_tf), dtype=tf.float32)

if DEBUG:
    small = 5000
    X_tf, y_tf = X_tf[:small], y_tf[:small]

# Preprocess features
new_X = preprocess_features(X_tf)
print(X_tf.shape)

print(y_tf.shape)
print(y_tf)

# Split data for training and validation
train_split = 0.7
train_idx = int(new_X.shape[0] * train_split)
# Apply the reshape function to the dataset using `map`
train_loader = (tf.data.Dataset.from_tensor_slices((new_X[:train_idx], y_tf[:train_idx]))
                .map(utils.reshape_X)  # Apply reshaping here
                .shuffle(buffer_size=1000)
                .batch(64))  # Adjust batch size as needed

val_loader = (tf.data.Dataset.from_tensor_slices((new_X[train_idx:], y_tf[train_idx:]))
              .map(utils.reshape_X)  # Apply reshaping here
              .batch(64))
# train_loader = tf.data.Dataset.from_tensor_slices((new_X[:train_idx], y_tf[:train_idx])).shuffle(buffer_size=1000).batch(64)
# val_loader = tf.data.Dataset.from_tensor_slices((new_X[train_idx:], y_tf[train_idx:])).batch(64)


# writing optuna results to a json file
results_file = "optuna_results.json"
optuna_results = []
#defining where to put data
storage = optuna.storages.RDBStorage(
    url="sqlite:///foo.sqlite3"
)

# Define the objective function
def objective(trial):
    # Hyperparameter to tune
    x = trial.suggest_categorical("x", [2, 4, 8, 12, 16])
    k = trial.suggest_categorical("k", [2, 4, 8, 12, 16])
    b = 16
    x_b3 = trial.suggest_categorical("x_b3", [2, 4, 8, 16])
    k_b3 = trial.suggest_categorical("k_b3", [2, 4, 8, 16])

    x_b4 = trial.suggest_categorical("x_b4", [2, 4, 8, 16])
    k_b4 = trial.suggest_categorical("k_b4", [2, 4, 8, 16])

    x_b5 = trial.suggest_categorical("x_b5", [2, 4, 8, 16])
    k_b5 = trial.suggest_categorical("k_b5", [2, 4, 8, 16])

    # Build the model with these six hyperparameters:
    sys_bits = SYS_BITS(x=x, k=k, b=b)
    model = UserModel(
        sys_bits = sys_bits, x_int_bits=0,
        sys_bits_b3=SYS_BITS(x=x_b3, k=k_b3, b=16),
        sys_bits_b4=SYS_BITS(x=x_b4, k=k_b4, b=16),
        sys_bits_b5=SYS_BITS(x=x_b5, k=k_b5, b=16),
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Training loop
    TRAINING_EPOCHS = 1
    total_train_loss = []

    for epoch in range(TRAINING_EPOCHS):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            X, y = batch
            # try:
            #     X = tf.convert_to_tensor(X.numpy().reshape((64, 2126, 1, 6)))
            # except ValueError:
            #     print("Skipping batch due to incompatible size")
            #     break

            # Training step
            with tf.GradientTape() as tape:
                out = model(X)
                loss = tf.reduce_mean(tf.keras.losses.MSE(out, y))

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss.numpy()

        # Average loss for the epoch
        total_loss /= len(train_loader)
        total_train_loss.append(total_loss)
        #print(f"Epoch {epoch + 1}, Average Loss: {total_loss}")

    # Return the minimum epoch loss as the objective value
    tf.keras.backend.clear_session()
    return min(total_train_loss)

#JSON FILE
def save_results_callback(study, trial):
    # Add the trial details to the results list
    trial_result = {
        "trial_number": trial.number,
        "value": trial.value,
        "params": trial.params
    }
    optuna_results.append(trial_result)

    # Save all results to JSON file after each trial
    with open(results_file, "w") as f:
        json.dump(optuna_results, f, indent=4)



y_tf = tf.convert_to_tensor(data_npz['y'], dtype=tf.float32)
target_scaler = MinMaxScaler((-1, 1))
y_tf = tf.convert_to_tensor(target_scaler.fit_transform(y_tf))
out_dim = y_tf.shape[-1]


@keras.saving.register_keras_serializable()
class UserModel(XModel):
   def __init__(self, sys_bits, x_int_bits, sys_bits_b3, sys_bits_b4, sys_bits_b5, *args, **kwargs):
       super().__init__(sys_bits, x_int_bits, *args, **kwargs)


       self.b0 = XBundle(
           # core=XDense(
           #    k_int_bits=0,
           #    b_int_bits=0,
           #    units=64,
           #    act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
           # )
           core=XConvBN(
               k_int_bits=0,
               b_int_bits=0,
               filters=64,
               kernel_size=1,
               act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
           ),
       )
      
       self.b1 = XBundle(
           core=XConvBN(
               k_int_bits=0,
               b_int_bits=0,
               filters=int(128/2),
               kernel_size=1,
               act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
           # core=XDense(
           #    k_int_bits=0,
           #    b_int_bits=0,
           #    units=int(128/dim_reduce_factor),
           #    act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)),
       )
      
       self.b2 = XBundle(
           core=XConvBN(
               k_int_bits=0,
               b_int_bits=0,
               filters=int(1024 / 2),
               kernel_size=1,
               act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
               ),
           pool=XPool(
               type='avg',
               pool_size=(2126,1),
               strides=(2126,1),
               padding='same',
               act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
           flatten=True
           # core=XDense(
           #    k_int_bits=0,
           #    b_int_bits=0,
           #    units=int(1024/dim_reduce_factor),
           #    act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None)),
       )


       self.b3 = XBundle(
           core=XDense(
               k_int_bits=0,
               b_int_bits=0,
               units=int(512 / 2),
               # units = out_dim,
               act=XActivation(sys_bits=sys_bits_b3, o_int_bits=0, type='relu', slope=0.125)
           ),
           # flatten=True
       )


       self.b4 = XBundle(
           core=XDense(
               k_int_bits=0,
               b_int_bits=0,
               units=int(128 / 2),
               act=XActivation(sys_bits=sys_bits_b4, o_int_bits=0, type='relu', slope=0.125)
           )
       )


       self.b5 = XBundle(
           core=XDense(
               k_int_bits=0,
               b_int_bits=0,
               units=out_dim,
               act=XActivation(sys_bits=sys_bits_b5, o_int_bits=0, type=None)),
           # flatten=True
       )


   def call (self, x):
       x = self.input_quant_layer(x)
       # print('input', x.shape)
       x = self.b0(x)
       # print(x.shape)
       x = self.b1(x)
       # print(x.shape)
       x = self.b2(x)
       # print(x.shape)
       # x = tf.keras.backend.sum(x, axis=1) / 2126
       # print(x.shape)
       x = self.b3(x)
       # print(x.shape)
       x = self.b4(x)
       # print(x.shape)
       x = self.b5(x)
       # print(f'Output from one pass: {x}')
       return x






# Set up and run the Optuna study
study = optuna.create_study(direction="minimize", storage=storage)  # Use "minimize" since we're minimizing loss
study.optimize(objective, n_trials=10, callbacks=[save_results_callback])

fig = optuna.visualization.plot_contour(study, params=["x", "k"])
fig.write_html("optuna_contour_plot.html")
# Output best parameters and validation loss
print(f"Best x value: {study.best_params['x']}")
print(f"Best validation loss: {study.best_value}")
print("Best trial:")
trial = study.best_trial
print(f"  Value (Min Loss): {trial.value}")
print("  Params:")
for key, value in trial.params.items():
   print(f"    {key}: {value}")