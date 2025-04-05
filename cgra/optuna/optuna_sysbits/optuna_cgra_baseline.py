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
from tqdm import tqdm
from time import time
from PointNet_merge import *
from read_point_cloud import * 
from utils import *

# from deepsocflow import *
from deepsocflow1.deepsocflow2.py.xbundle import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import pickle
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
import utils1

import scipy
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm, chisquare
import numpy as np
import joblib
import optuna

(SIM, SIM_PATH) = ('xsim', "F:/Xilinx/Vivado/2022.2/bin/") if os.name=='nt' else ('verilator', '')
np.random.seed(42)

# print("Using xbundle from:", inspect.getfile(XBundle))


'''
Define Model
'''

# sys_bits = SYS_BITS(x=8, k=8, b=16)
NB_EPOCH = 2
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
# TRAINING_EPOCHS = 30
DEBUG = False
training = True

# Load and preprocess data outside of the objective function
pmtxyz = get_pmtxyz("/home/amehta/PointNET_PMT_keras/data/pmt_xyz.dat")
data_npz = np.load('/home/amehta/PointNET_PMT_keras/data/train_X_y_ver_all_xyz_energy.npz')
X_tf = tf.convert_to_tensor(data_npz['X'], dtype=tf.float32)
y_tf = tf.convert_to_tensor(data_npz['y'], dtype=tf.float32)
if DEBUG:
    small = 5000
    X_tf, y_tf = X_tf[:small], y_tf[:small]

# Scale target data
target_scaler = MinMaxScaler((-1, 1))
y_tf = target_scaler.fit_transform(y_tf)
y_tf = tf.convert_to_tensor(y_tf)

# Preprocess features
new_X = preprocess_features(X_tf)

# Define the reshape function
def reshape_X(X, y):
    X = tf.reshape(X, (2126, 1, 6))  
    return X, y

joblib.dump(target_scaler, 'target_scaler.gz')

train_split = 0.7
val_split = 0.3
train_idx = int(new_X.shape[0] * train_split)
val_idx = int(train_idx + new_X.shape[0] * val_split)
train = tf.data.Dataset.from_tensor_slices((new_X[:train_idx], y_tf[:train_idx]))
val = tf.data.Dataset.from_tensor_slices((new_X[train_idx:val_idx], y_tf[train_idx:val_idx]))
test = tf.data.Dataset.from_tensor_slices((new_X[val_idx:], y_tf[val_idx:]))
train_loader = train.shuffle(buffer_size=len(new_X)).batch(BATCH_SIZE)
val_loader = val.batch(BATCH_SIZE)
test_loader = val.batch(BATCH_SIZE)
print(f"num. total: {len(new_X)} train: {len(train)}, val: {len(val)}, test: {len(test)}")
print(pmtxyz.shape, tf.shape(new_X), y_tf.shape)

n_data, _, F_dim = X_tf.shape
dim = F_dim
dim_reduce_factor = 2
out_dim = y_tf.shape[-1] # 4
dimensions = dim
nhits = 2126
encoder_input_shapes = [dimensions, 64, int(128 / dim_reduce_factor)]
(_, F1, F2), latent_dim = encoder_input_shapes, int(1024 / dim_reduce_factor)
decoder_input_shapes = latent_dim, int(512/dim_reduce_factor), int(128/dim_reduce_factor)
latent_dim, F3, F4 = decoder_input_shapes
print("Test", F1, F2, dim, dim_reduce_factor, out_dim, dimensions)

####
#Optuna
# writing optuna results to a json file
results_file = "optuna_results.json"
optuna_results = []
#defining where to put data
storage = optuna.storages.RDBStorage(
    url="sqlite:///foo.sqlite3"
)
####

def objective(trial):
    # Tune sys bits for each layer independently
    x_b0 = trial.suggest_categorical("x_b0", [2, 4, 8, 12, 16])
    k_b0 = trial.suggest_categorical("k_b0", [2, 4, 8, 12, 16])

    x_b1 = trial.suggest_categorical("x_b1", [2, 4, 8, 12, 16])
    k_b1 = trial.suggest_categorical("k_b1", [2, 4, 8, 12, 16])

    x_b2 = trial.suggest_categorical("x_b2", [2, 4, 8, 12, 16])
    k_b2 = trial.suggest_categorical("k_b2", [2, 4, 8, 12, 16])

    x_b3 = trial.suggest_categorical("x_b3", [2, 4, 8, 12, 16])
    k_b3 = trial.suggest_categorical("k_b3", [2, 4, 8, 12, 16])
    
    x_b4 = trial.suggest_categorical("x_b4", [2, 4, 8, 12, 16])
    k_b4 = trial.suggest_categorical("k_b4", [2, 4, 8, 12, 16])
    
    x_b5 = trial.suggest_categorical("x_b5", [2, 4, 8, 12, 16])
    k_b5 = trial.suggest_categorical("k_b5", [2, 4, 8, 12, 16])

    sys_bits_b0 = SYS_BITS(x=x_b0, k=k_b0, b=16)
    sys_bits_b1 = SYS_BITS(x=x_b1, k=k_b1, b=16)
    sys_bits_b2 = SYS_BITS(x=x_b2, k=k_b2, b=16)
    sys_bits_b3 = SYS_BITS(x=x_b3, k=k_b3, b=16)
    sys_bits_b4 = SYS_BITS(x=x_b4, k=k_b4, b=16)
    sys_bits_b5 = SYS_BITS(x=x_b5, k=k_b5, b=16)

    sys_bits_b0 = SYS_BITS(x=x_b0, k=k_b0, b=16)
    sys_bits_b1 = SYS_BITS(x=x_b1, k=k_b1, b=16)
    sys_bits_b2 = SYS_BITS(x=x_b2, k=k_b2, b=16)
    sys_bits_b3 = SYS_BITS(x=x_b3, k=k_b3, b=16)
    sys_bits_b4 = SYS_BITS(x=x_b4, k=k_b4, b=16)
    sys_bits_b5 = SYS_BITS(x=x_b5, k=k_b5, b=16)

    # WILL USE LATER #
    # Tune kernel sizes for convolutional layers
    # kernel_size_b0 = trial.suggest_categorical("kernel_size_b0", [1, 3, 5, 7])
    # kernel_size_b1 = trial.suggest_categorical("kernel_size_b1", [1, 3, 5])
    # kernel_size_b2 = trial.suggest_categorical("kernel_size_b2", [1, 3, 5])
    
    # # # Optionally tune filters for conv layers
    # filters_b0 = trial.suggest_categorical("filters_b0", [32, 64, 128])
    # filters_b1 = trial.suggest_categorical("filters_b1", [64, 128, 256])
    # filters_b2 = trial.suggest_categorical("filters_b2", [128, 256, 512])
    
    # # Tune dense units for blocks b3 and b4
    # units_b3 = trial.suggest_categorical("units_b3", [64, 128, 256])
    # units_b4 = trial.suggest_categorical("units_b4", [32, 64, 128])

    input_shape = X_tf.shape[1:]
    x = x_in =  Input((pmtxyz.shape[0], 1, 6), name="input")
    user_model =UserModel(
        sys_bits_b0=sys_bits_b0,
        x_int_bits=0,
        sys_bits_b1=sys_bits_b1,
        sys_bits_b2=sys_bits_b2,
        sys_bits_b3=sys_bits_b3,
        sys_bits_b4=sys_bits_b4,
        sys_bits_b5=sys_bits_b5,
        # kernel_sizes={"b0": kernel_size_b0, "b1": kernel_size_b1, "b2": kernel_size_b2},
        # filters={"b0": filters_b0, "b1": filters_b1, "b2": filters_b2},
        # units={"b3": units_b3, "b4": units_b4}
    )
    x = user_model(x_in)
    model = Model(inputs=[x_in], outputs=[x])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    TRAINING_EPOCHS = 7
    total_train_loss = []
    total_val_loss = 0

    for epoch in range(TRAINING_EPOCHS):
        total_loss = 0
        for batch in train_loader:
            X, y = batch
            try:
                X = tf.convert_to_tensor(X.numpy().reshape((64, 2126, 1, 6)))
            except ValueError:
                print("Skipping batch due to incompatible size")
                continue

            with tf.GradientTape() as tape:
                out = model(X)
                loss = tf.reduce_mean(tf.keras.losses.MSE(out, y))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss.numpy()
        
        total_loss /= len(train_loader)
        total_train_loss.append(total_loss)
        trial.report(total_loss, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()

    tf.keras.backend.clear_session()

    for batch in val_loader:
        X, y = batch
        try:
            X = tf.convert_to_tensor(X.numpy().reshape((BATCH_SIZE, 2126, 1, 6)))
        except ValueError:
            print("Skipping batch due to incompatible size")
            break
        out = model(X)
        out = tf.convert_to_tensor(target_scaler.inverse_transform(out))
        y = tf.convert_to_tensor(target_scaler.inverse_transform(y))
        val_loss = tf.reduce_mean(tf.keras.losses.MSE(out, y))
        total_val_loss += val_loss.numpy()

    return total_val_loss / len(val_loader)

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
    def __init__(self, sys_bits_b0, x_int_bits, sys_bits_b1, sys_bits_b2, 
                 sys_bits_b3, sys_bits_b4, sys_bits_b5, *args, **kwargs):
        super().__init__(sys_bits_b0, x_int_bits, *args, **kwargs)
        
        # Save hyperparameters for later use
        self.sys_bits_b0 = sys_bits_b0
        self.sys_bits_b1 = sys_bits_b1
        self.sys_bits_b2 = sys_bits_b2
        self.sys_bits_b3 = sys_bits_b3
        self.sys_bits_b4 = sys_bits_b4
        self.sys_bits_b5 = sys_bits_b5

        self.b0 = XBundle(
            core=XConvBN(
                k_int_bits=0,
                b_int_bits=0,
                filters=64,
                kernel_size=1,
                act=XActivation(sys_bits=self.sys_bits_b0, o_int_bits=0, type='relu', slope=0)
            ),
        )
      
        self.b1 = XBundle(
            core=XConvBN(
                k_int_bits=0,
                b_int_bits=0,
                filters=64,
                kernel_size=1,
                act=XActivation(sys_bits=self.sys_bits_b1, o_int_bits=0, type='relu', slope=0),
            ),
        )
      
        self.b2 = XBundle(
            core=XConvBN(
                k_int_bits=0,
                b_int_bits=0,
                filters=216,
                kernel_size=1,
                act=XActivation(sys_bits=self.sys_bits_b2, o_int_bits=0, type='relu', slope=0)
            ),
            pool=XPool(
                type='avg',
                pool_size=(2126, 1),
                strides=(2126, 1),
                padding='same',
                act=XActivation(sys_bits=self.sys_bits_b2, o_int_bits=0, type=None),
            ),
            flatten=True
        )

        self.b3 = XBundle(
            core=XDense(
                k_int_bits=0,
                b_int_bits=0,
                units=int(512 / 2),
                act=XActivation(sys_bits=self.sys_bits_b3, o_int_bits=0, type='relu', slope=0.125)
            ),
        )

        self.b4 = XBundle(
            core=XDense(
                k_int_bits=0,
                b_int_bits=0,
                units=int(128 / 2),
                act=XActivation(sys_bits=self.sys_bits_b4, o_int_bits=0, type='relu', slope=0.125)
            )
        )

        self.b5 = XBundle(
            core=XDense(
                k_int_bits=0,
                b_int_bits=0,
                units=out_dim,
                act=XActivation(sys_bits=self.sys_bits_b5, o_int_bits=0, type=None)
            ),
        )

    def call(self, x):
        x = self.input_quant_layer(x)
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return x



# Set up and run the Optuna study
study = optuna.create_study(study_name="baseline_sys_bits_per_layer(hopefully_final)", direction="minimize", storage=storage)  # Minimizing loss
study.optimize(objective, n_trials=10, callbacks=[save_results_callback])

# # Contour plot for kernel sizes
# fig_kernel = optuna.visualization.plot_contour(
#     study, params=["kernel_size_b0", "kernel_size_b1", "kernel_size_b2"]
# )
# fig_kernel.write_html("optuna_contour_kernel_sizes.html")

# # Contour plot for filters
# fig_filters = optuna.visualization.plot_contour(
#     study, params=["filters_b0", "filters_b1", "filters_b2"]
# )
# fig_filters.write_html("optuna_contour_filters.html")

# # Contour plot for dense units
# fig_units = optuna.visualization.plot_contour(
#     study, params=["units_b3", "units_b4"]
# )
# fig_units.write_html("optuna_contour_units.html")

# Parallel coordinate plot for kernel sizes, filters, and dense units together
# fig_parallel = optuna.visualization.plot_parallel_coordinate(
#     study,
#     params=["kernel_size_b0", "kernel_size_b1", "kernel_size_b2",
#             "filters_b0", "filters_b1", "filters_b2",
#             "units_b3", "units_b4"]
# )
# fig_parallel.write_html("optuna_parallel_coordinate.html")

# Optionally, you could also plot sys bits parameters if desired:
# fig_sys_bits = optuna.visualization.plot_parallel_coordinate(
#     study,
#     params=["x_b0", "k_b0", "x_b1", "k_b1", "x_b2", "k_b2",
#             "x_b3", "k_b3", "x_b4", "k_b4", "x_b5", "k_b5"]
# )
# fig_sys_bits.write_html("optuna_parallel_sys_bits.html")

##################
## Contour plot ##
##################
layers = ["b0", "b1", "b2", "b3", "b4", "b5"]

for layer in layers:
    # Construct the parameter names for x and k
    param_x = f"x_{layer}"
    param_k = f"k_{layer}"
    
    # Create the contour plot for the given layer's x and k
    fig = optuna.visualization.plot_contour(study, params=[param_x, param_k])
    
    # Save the plot as an HTML file
    fig.write_html(f"baseline_optuna_contour_{layer}.html")


# Output best trial details
print("Best trial:")
print(f"  Value (Min Loss): {study.best_trial.value}")
print("  Params:")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")