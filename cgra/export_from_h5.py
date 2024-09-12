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
import wandb
from tqdm import tqdm
from time import time
from PointNet_merge import *
from read_point_cloud import * 
from utils import *

from deepsocflow import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib

dim_reduce_factor = 2
out_dim = 4
@keras.saving.register_keras_serializable()
class UserModel(XModel):
    def __init__(self, sys_bits, x_int_bits, *args, **kwargs):
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
                filters=int(128/dim_reduce_factor),
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
                filters=int(1024 / dim_reduce_factor),
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
                units=int(512 / dim_reduce_factor),
                # units = out_dim,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0.125)
            ),
            # flatten=True
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
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None)),
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

loaded_model = load_qmodel("mnist.h5")
# model.save("mnist.keras")
# loaded_model = tf.keras.saving.load_model("mnist.keras")

#score = loaded_model.evaluate(test_loader, verbose=0)
#print(f"Test loss:{score[0]}, Test accuracy:{score[1]}")
import pickle
with open(f'X.pickle', 'rb') as handle:
    global custom_input
    custom_input = pickle.load(handle)


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
                                        max_batch_size       = [ 128      ], 
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
    export_inference(loaded_model, hw, custom_input=deploy_val_X)
    # verify_inference(loaded_model, hw, SIM=SIM, SIM_PATH=SIM_PATH)

    d_perf = predict_model_performance(hw)
    pp = pprint.PrettyPrinter(indent=4)
    print(f"Predicted Performance")
    pp.pprint(d_perf)
