from deepsocflow1.deepsocflow2.py.xbundle import *
# from deepsocflow.py import XActivation
# from deepsocflow.py import XConvBN
# from deepsocflow.py import XBundle
# from deepsocflow.py import XPool
# from deepsocflow.py import XDense
# from deepsocflow.py import XModel

# ###
# #need to find a way to get this
# out_dim = y_tf.shape[-1] # 4
# ###

class UserModel(XModel):

    def __init__(self, sys_bits, x_int_bits,  out_dim, *args, **kwargs):

        super().__init__(sys_bits, x_int_bits, *args, **kwargs)
        
        self.dim_reduce_factor = 2

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
                filters=int(128/self.dim_reduce_factor),
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
                filters=int(1024 / self.dim_reduce_factor),
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
                units=int(512 / self.dim_reduce_factor),
                # units = out_dim,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0.125)
            ),
            # flatten=True
        )

        self.b4 = XBundle( 
            core=XDense(
                k_int_bits=0,
                b_int_bits=0,
                units=int(128 / self.dim_reduce_factor),
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
