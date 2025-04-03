# import joblib
# from sklearn.preprocessing import MinMaxScaler
# from PointNet_merge import *
# from read_point_cloud import *
# from utils import *

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

from deepsocflow import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import pickle
from optimize_cgra.usermodel import *
from optimize_cgra.util import *
from optimize_cgra.train import *

def main():
    (SIM, SIM_PATH) = ('xsim', "F:/Xilinx/Vivado/2022.2/bin/") if os.name=='nt' else ('verilator', '')
    np.random.seed(42)
    tf.random.set_seed(42)

    DEBUG = False
    BATCH_SIZE = 64

    #parameters
    sys_bits = SYS_BITS(x=8, k=8, b=16)

    #loads in data
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

    new_X = preprocess_features(X_tf)
    print(X_tf.shape)

    print(y_tf.shape)
    print(y_tf)

    y_tf = target_scaler.fit_transform(y_tf)
    y_tf = tf.convert_to_tensor(y_tf, dtype=tf.float32)


    #train, val split
    train_split = 0.7
    val_split = 0.3

    #idx for train and val
    train_idx = int(new_X.shape[0] * train_split)
    val_idx = train_idx + int(new_X.shape[0] * val_split)   

    train = tf.data.Dataset.from_tensor_slices((new_X[:train_idx], y_tf[:train_idx]))
    val = tf.data.Dataset.from_tensor_slices((new_X[train_idx:val_idx], y_tf[train_idx:val_idx]))
    test = tf.data.Dataset.from_tensor_slices((new_X[val_idx:], y_tf[val_idx:]))
    train_loader = train.shuffle(buffer_size=len(new_X)).batch(BATCH_SIZE)
    val_loader = val.batch(BATCH_SIZE)
    test_loader = val.batch(BATCH_SIZE)
    print(f"num. total: {len(new_X)} train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print(pmtxyz.shape, tf.shape(new_X), y_tf.shape)

    #have no idea what this does ngl

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


    ###
    #THIS IS CALLING THE MODEL
    input_shape = X_tf.shape[1:]
    x = x_in =  Input((pmtxyz.shape[0], 1, 6), name="input")
    user_model = UserModel(sys_bits=sys_bits, x_int_bits=0, out_dim = out_dim)
    x = user_model(x_in)
    model = Model(inputs=[x_in], outputs=[x])
    ###

    ####
    ####
    #Summarizing Model
    print(model.submodules)
    #print(y[:5], model(X_tf[:5]))
    for layer in model.submodules:
        try:
            print(layer.summary())
            for w, weight in enumerate(layer.get_weights()):
                    print(layer.name, w, weight.shape)
        except:
            pass

    print(summary_plus(model))
    model.summary(expand_nested=True)
    ####
    ####

    traina(model, train_loader, target_scaler, val_loader)

    # ## INFERENCE
    # eval(model,
    #     val_loader, 64, target_scaler)
    
    # save_model(Model, 'cgra_model.h5')
    


if __name__ == '__main__':
    main()