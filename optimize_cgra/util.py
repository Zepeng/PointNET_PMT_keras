import tensorflow as tf
from keras.models import Model, save_model
import pytest
import itertools
import matplotlib
from matplotlib import *
import pickle
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

# def reshape_X(X, y):
#     # Reshape X to the desired shape
#     X = tf.reshape(X, (2126, 1, 6))  # Adjust based on your specific requirements
#     return X, y

def reshape_X(X, y):
    # Reshape X to the desired shape
    X = tf.reshape(X, (2126, 1, 6))  # Adjust based on your specific requirements
    return X, y

def plot_and_save(model, abs_diff, diff, dist, total_val_loss):
    abs_x_diff, abs_y_diff, abs_z_diff, abs_energy_diff = tf.reduce_mean(abs_diff, axis=0)
    energy_diff = tf.concat(diff["energy"], axis=0).cpu()
    energy_pred = tf.concat(dist["energy_pred"], axis=0).cpu()
    energy = tf.concat(dist["energy"], axis=0).cpu()

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

    with open(f'cgra_pointnet.pickle', 'wb') as handle:
        pickle.dump(val_save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.close()
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 15))
    fig.suptitle(f"Val. MSE: {total_val_loss:.2f} (MSE(x) + MSE(y) + MSE(y) + MSE(energy))\n\
    Avg. abs. diff. in x={abs_x_diff:.2f}, y={abs_y_diff:.2f}, z={abs_z_diff:.2f}, energy={abs_energy_diff:.2f}", fontsize=20)

    ## diff. plots
    x_diff_range = (-50, 50)
    large_fontsize = 20
    axes[0,0].hist(x_diff, bins=20, range=x_diff_range, edgecolor='black')
    axes[0,0].set_title(r"x_diff ($x - \hat{x}$)", fontsize=large_fontsize)
    axes[0,0].set_xlabel('x diff', fontsize=large_fontsize)
    axes[0,0].set_ylabel('Probability', fontsize=large_fontsize)

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
    axes[1,0].set_xlabel('x (cm)', fontsize=large_fontsize)
    axes[1,0].set_ylabel('Probability', fontsize=large_fontsize)

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
    axes[1,2].set_xlabel(r'z (cm)', fontsize=large_fontsize)
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

    plt.savefig(f'./cgra_pointNET_last_hist.png')
    plt.close()

    ###
    #THINK I SHOULD MOVE THIS
    ###
    save_model(model, 'cgra_model.h5')

def summary_plus(layer, i=0):
    if hasattr(layer, 'layers'):
        if i != 0: 
            layer.summary()
        for l in layer.layers:
            i += 1
            summary_plus(l, i=i)

### I have no idea where to put this or what it does # rip
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
    export_inference(model, hw, custom_input=deploy_val_X)
    # verify_inference(loaded_model, hw, SIM=SIM, SIM_PATH=SIM_PATH)

    d_perf = predict_model_performance(hw)
    pp = pprint.PrettyPrinter(indent=4)
    print(f"Predicted Performance")
    pp.pprint(d_perf)
