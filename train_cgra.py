import os
import pytest
import itertools
import sys
sys.path.append("../../")
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
import wandb
from tqdm import tqdm
from time import time
from PointNet_merge import *
from read_point_cloud import * 
from utils import *

from deepsocflow import *
from sklearn.preprocessing import MinMaxScaler


(SIM, SIM_PATH) = ('xsim', "F:/Xilinx/Vivado/2022.2/bin/") if os.name=='nt' else ('verilator', '')
np.random.seed(42)

'''
Dataset
'''



'''
Define Model
'''

sys_bits = SYS_BITS(x=8, k=8, b=16)
NB_EPOCH = 2
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
TRAINING_EPOCHS = 20
DEBUG = True

pmtxyz = get_pmtxyz("/home/amigala/PointNET_PMT_keras/data/pmt_xyz.dat")
data_npz = np.load('/home/amigala/PointNET_PMT_keras/data/train_X_y_ver_all_xyz_energy.npz')
X_tf = tf.convert_to_tensor(data_npz['X'], dtype=tf.float32)
y_tf = tf.convert_to_tensor(data_npz['y'], dtype=tf.float32)
if DEBUG:
    small = 5000
    X_tf, y_tf = X_tf[:small], y_tf[:small]

new_X = preprocess_features(X_tf)

# min/max scale the training data and the target data using their own scalers
# training_scaler = MinMaxScaler((-1,1))
# # print(new_X.shape)
# original_shape = new_X.shape
# new_X = new_X.numpy().reshape(new_X.shape[0], new_X.shape[1]*new_X.shape[2])
# # print(new_X.shape)
# # print(new_X)
# new_X = training_scaler.fit_transform(new_X).reshape(original_shape)
# new_X = tf.convert_to_tensor(new_X)

# now scale the target
target_scaler = MinMaxScaler((-1,1))
# print(y_tf.shape)
# y_tf_original_shape = y_tf.shape
# y_tf = y_tf.numpy().reshape(y_tf.shape[0], y_tf.shape[1]*y_tf.shape[2])
print(y_tf.shape)
print(y_tf)
y_tf = y_tf.numpy()
y_tf[:,3] *= 160 # scale the energy by 160 before fitting
print(tf.convert_to_tensor(y_tf))
y_tf = target_scaler.fit_transform(y_tf)
y_tf = tf.convert_to_tensor(y_tf)
# print(y_tf)
# assert 0

train_split = 0.7
val_split = 0.3
train_idx = int(new_X.shape[0] * train_split)
val_idx = int(train_idx + new_X.shape[0] * train_split)
train = tf.data.Dataset.from_tensor_slices((new_X[:train_idx], y_tf[:train_idx]))
val = tf.data.Dataset.from_tensor_slices((new_X[train_idx:val_idx], y_tf[train_idx:val_idx]))
test = tf.data.Dataset.from_tensor_slices((new_X[val_idx:], y_tf[val_idx:]))
train_loader = train.shuffle(buffer_size=len(new_X)).batch(BATCH_SIZE)
val_loader = val.batch(BATCH_SIZE)
test_loader = val.batch(BATCH_SIZE)
print(f"num. total: {len(new_X)} train: {len(train)}, val: {len(val)}, test: {len(test)}")
print(pmtxyz.shape, tf.shape(new_X), y_tf.shape)

# input_shape = (2126, 1, 6)#X_tf.shape[1:]
# n_hits, _, F_dim = input_shape#X_tf.shape
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

@keras.saving.register_keras_serializable()
class UserModel(XModel):
    def __init__(self, sys_bits, x_int_bits, *args, **kwargs):
        super().__init__(sys_bits, x_int_bits, *args, **kwargs)

        self.b0 = XBundle( 
            # core=XConvBN(
            #     k_int_bits=0,
            #     b_int_bits=0,
            #     filters=64,
            #     kernel_size=1,
            #     act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)),
            core=XDense(
               k_int_bits=0,
               b_int_bits=0,
               units=64,
               act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
               ),
            )
        
        self.b1 = XBundle( 
            # core=XConvBN(
            #     k_int_bits=0,
            #     b_int_bits=0,
            #     filters=int(128/dim_reduce_factor),
            #     kernel_size=1,
            #     act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
            core=XDense(
               k_int_bits=0,
               b_int_bits=0,
               units=int(128/dim_reduce_factor),
               act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)),
        )
        
        self.b2 = XBundle( 
            # core=XConvBN(
            #     k_int_bits=0,
            #     b_int_bits=0,
            #     filters=int(1024 / dim_reduce_factor),
            #     kernel_size=1,
            #     act=XActivation(sys_bits=sys_bits, o_int_bits=0, type="relu", slope=1)
            #     ),
            # pool=XPool(
            #     type='avg',
            #     pool_size=(2126,1),
            #     strides=(2126,1),
            #     padding='same',
            #     act=XActivation(sys_bits=sys_bits, o_int_bits=0, type="relu", slope=1),),
            # flatten=True
            core=XDense(
               k_int_bits=0,
               b_int_bits=0,
               units=int(1024/dim_reduce_factor),
               act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None)),
        )

        self.b3 = XBundle( 
            core=XDense(
                k_int_bits=0,
                b_int_bits=0,
                units=int(512 / dim_reduce_factor),
                # units = out_dim,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0.125)
            ),
            flatten=True
        )

        self.b4 = XBundle( 
            core=XDense(
                k_int_bits=0,
                b_int_bits=0,
                units=int(128 / dim_reduce_factor),
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0.125)
            )
        )

        self.b5 = XBundle(
            core=XDense(
                k_int_bits=0,
                b_int_bits=0,
                units=out_dim,
                act=XActivation(sys_bits=SYS_BITS(x=32, k=32, b=32), o_int_bits=0, type=None)),
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
        x = tf.keras.backend.sum(x, axis=1) / 2126
        # print(x.shape)
        x = self.b3(x)
        # print(x.shape)
        x = self.b4(x)
        # print(x.shape)
        x = self.b5(x)
        # print(f'Output from one pass: {x}')
        return x

input_shape = X_tf.shape[1:]
# (pmtxyz.shape[0], tf.shape(new_X)[2])
# (pmtxyz.shape[0], 1, tf.shape(new_X)[2])
x = x_in =  Input((pmtxyz.shape[0], tf.shape(new_X)[2]), name="input")
user_model = UserModel(sys_bits=sys_bits, x_int_bits=0)
x = user_model(x_in)

model = Model(inputs=[x_in], outputs=[x])

'''
Summarize model
'''
print(model.submodules)
#print(y[:5], model(X_tf[:5]))
for layer in model.submodules:
    try:
        print(layer.summary())
        for w, weight in enumerate(layer.get_weights()):
                print(layer.name, w, weight.shape)
    except:
        pass
# print_qstats(model.layers[1])

def summary_plus(layer, i=0):
    if hasattr(layer, 'layers'):
        if i != 0: 
            layer.summary()
        for l in layer.layers:
            i += 1
            summary_plus(l, i=i)

print(summary_plus(model)) # OK 
model.summary(expand_nested=True)
# assert 0

# '''
# Train Model
# '''
# model.compile(loss="mse", optimizer=Adam(learning_rate=0.0001), metrics=["mse"])
#history = model.fit(
#        train_loader,
#        #x_train, 
#        #y_train, 
#        batch_size=BATCH_SIZE,
#        epochs=NB_EPOCH, 
#        #initial_epoch=1, 
#        verbose=True,
#        )

'''
Train
'''

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
epochs = range(TRAINING_EPOCHS)
#model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
model.compile(optimizer=optimizer)

## Train Loop
# wandb.login()
# run = wandb.init(
#     # Set the project where this run will be logged
#     project="pointnet-cgra"
#     # Track hyperparameters and run metadata
#     # config={
#     #     "learning_rate": lr,
#     #     "epochs": epochs,
#     # }
# )

# Initialize tqdm progress bar
pbar = tqdm(total=TRAINING_EPOCHS, mininterval=10)
# Initialize best validation loss and best train loss
best_val, best_train = float("inf"), float("inf")
tot_train_lst = []

# Loop through epochs
for epoch in range(TRAINING_EPOCHS):
    total_loss = 0
    
    # Loop through batches in training loader
    for i, batch in enumerate(train_loader):
        X, y = batch
        # try:
        #     X = tf.convert_to_tensor(X.numpy().reshape((BATCH_SIZE, 2126, 1, 6)))
        # except ValueError:
        #     print("skipping batch due to incompatible size")
        #     break
        # for some reason, this reshape can't always work (i'm guessing it's an issue at the
        # end of the data set where there is less than a batch's worth of data)
        
        # Forward 
        index = 0
        with tf.GradientTape() as tape:
            # need to reshape data here to bypass training issues
            # print(X.shape)
            # assert 0
            out = model(X)
            # print(out)
            # print(out.shape)
            energy_mult = 160  # scale energy loss
            # Scale the last dimension of 'out' and 'y' tensors
            # out = tf.convert_to_tensor(target_scaler.inverse_transform(out.numpy()))
            # y = tf.convert_to_tensor(target_scaler.inverse_transform(y.numpy()))
            # print(out.shape)
            # print(y.shape)

            # out = tf.concat([out[:, :-1], energy_mult * tf.expand_dims(out[:, -1], axis=-1)], axis=-1)
            # y = tf.concat([y[:, :-1], energy_mult * tf.expand_dims(y[:, -1], axis=-1)], axis=-1)
            # print(f'Output from one pass: {out.numpy()}\n y_true: {y.numpy()}')
            # assert 0
            # out = tf.convert_to_tensor(target_scaler.transform(out.numpy()))
            # y = tf.convert_to_tensor(target_scaler.transform(y.numpy()))

            #out[:, -1], y[:, -1] = energy_mult * out[:, -1], energy_mult * y[:, -1]
            loss = tf.reduce_mean(tf.keras.losses.MSE(out, y))
            if index % 100 == 0:
                out = tf.convert_to_tensor(target_scaler.inverse_transform(out.numpy()))
                y = tf.convert_to_tensor(target_scaler.inverse_transform(y.numpy()))
                # print(out)
                # print(y)
            # print(type(loss))
            # assert 0
        
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
    print(total_loss)
    
    # Adjust learning rate based on training loss
    #prev_lr = optimizer.learning_rate.numpy()
    #reduce_lr.on_epoch_end(epoch, logs={'loss': total_loss})
    #scheduler.step(total_loss)
    
    # Validation every 10 epochs or last epoch
    # if epoch == TRAINING_EPOCHS - 1:
    #     total_val_loss = 0
        
    #     # Validation (evaluation mode)
    #     for batch in val_loader:
    #         X, y = batch
    #         out = model(X)
    #         val_loss = tf.reduce_mean(tf.keras.losses.MSE(out, y))
    #         total_val_loss += val_loss.numpy()
        
    #     # Calculate average validation loss
    #     total_val_loss /= len(val_loader)
        
    #     # Update progress bar
    #     # pbar.update(10)
        
    #     # Save best model based on validation loss
    #     if total_val_loss < best_val:
    #         best_val = total_val_loss
    #         if args.save_best:
    #             model.save(save_name + ".keras")
    
    # Log training loss
    # else:
        # print(total_loss)
    
# Close progress bar
pbar.close()

## time
# tot_time = time() - strt

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
save_name = f'./cgra_pointNET_last'
import argparse
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
args = parser.parse_args()

plot_reg(diff=diff, dist=dist, total_val_loss=total_val_loss, abs_diff=abs_diff, save_name=save_name, args=args)


'''
Save & Reload
'''

# save_model(model, "mnist.h5")
# loaded_model = load_qmodel("mnist.h5")

#score = loaded_model.evaluate(test_loader, verbose=0)
#print(f"Test loss:{score[0]}, Test accuracy:{score[1]}")




def product_dict(**kwargs):
    for instance in itertools.product(*(kwargs.values())):
        yield dict(zip(kwargs.keys(), instance))

@pytest.mark.parametrize("PARAMS", list(product_dict(
                                        processing_elements  = [(16,32)   ],
                                        frequency_mhz        = [ 250     ],
                                        bits_input           = [ 8       ],
                                        bits_weights         = [ 8       ],
                                        bits_sum             = [ 32      ],
                                        bits_bias            = [ 16      ],
                                        max_batch_size       = [ 64      ], 
                                        max_channels_in      = [ 2048    ],
                                        max_kernel_size      = [ 9       ],
                                        max_image_size       = [ 2126    ],
                                        max_n_bundles        = [ 64      ],
                                        ram_weights_depth    = [ 20      ],
                                        ram_edges_depth      = [ 288     ],
                                        axi_width            = [ 128      ],
                                        config_baseaddr      = ["B0000000"],
                                        target_cpu_int_bits  = [ 32       ],
                                        valid_prob           = [ 1     ],
                                        ready_prob           = [ 1     ],
                                        data_dir             = ['vectors'],
                                    )))
def test_dnn_engine(PARAMS):

    '''
    SPECIFY HARDWARE
    '''
    hw = Hardware (**PARAMS)
    hw.export_json()
    hw = Hardware.from_json('hardware.json')
    hw.export() # Generates: config_hw.svh, config_hw.tcl
    hw.export_vivado_tcl(board='zcu104')


    '''
    VERIFY & EXPORT
    '''
    # export_inference(loaded_model, hw, hw.ROWS)
    # verify_inference(loaded_model, hw, SIM=SIM, SIM_PATH=SIM_PATH)

    # d_perf = predict_model_performance(hw)
    # pp = pprint.PrettyPrinter(indent=4)
    # print(f"Predicted Performance")
    # pp.pprint(d_perf)
