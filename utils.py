import pandas as pd
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
# from pprint import pprint

def init_logfile(i):
    '''
        create and set logfile to be written. Also write init messages such as args and seed
    '''
    save_dir = f"./{i}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_file = open(f"{i}/train.txt", 'w', buffering=1)
    sys.stderr = log_file
    sys.stdout = log_file
    return save_dir, log_file

def clean_nohup_out():
    with open("nohup.out", "w") as f:
        pass

def preprocess_features(X):
    '''
        switch to match Aobo's syntax (time, charge, x, y, z) -> (x, y, z, label, time, charge)
        insert "label" feature to tensor. This feature (0 or 1) is the activation of sensor
    '''
    index = tf.constant([2, 3, 4, 0, 1])
    X = tf.gather(X, index, axis=-1)

    #X = X[:, :, [2, 3, 4, 0, 1]]
    #X = X.mT
    X = tf.transpose(X)
    print("preprocessing data...", X.shape)

    ## insert "label" feature to tensor. This feature (0 or 1) is the activation of sensor
    #new_X = torch.zeros(X.shape[0], X.shape[1]+1, X.shape[2])
    #new_X = tf.zeros([X.shape[0], X.shape[1]+1, X.shape[2]], dtype=X.dtype)
    mask_3 = tf.cast(tf.not_equal(X[3, :, :], 0), dtype=tf.bool)
    mask_4 = tf.cast(tf.not_equal(X[4, :, :], 0), dtype=tf.bool)
    label_feat = tf.cast(tf.logical_and(mask_3, mask_4), dtype=X.dtype)
    #label_feat = ((X[:, 3, :] != 0) & (X[:, 4, :] != 0)).float() ## register 0 if both time and charge == 0.
    #new_X[:, 3, :] = label_feat
    #new_X[:, :3, :] = X[:, :3, :]
    #new_X[:, 4:, :] = X[:, 3:, :]
    X = tf.concat([X[:3, :, :], tf.expand_dims(label_feat, axis=0), X[3:, :, :]], axis=0)
    X = tf.transpose(X, perm=[2, 1, 0])
    print(f"Training Data shape: {X.shape}")

    #return tf.transpose(X,perm=[2, 1, 0]),  tf.transpose(new_X, perm=[2, 1, 0])
    return X

def plot_r_z(diff, dist, abs_diff, total_val_loss, save_name):
    z_pure_diff, radius_pure_diff = tf.reduce_mean(abs_diff, axis=0)
    tot_z_pure_diff, tot_r_pure_diff = abs_diff.sum(axis=0)

    z_diff = tf.concat(diff["z"], axis=0).cpu()
    radius_diff = tf.concat(diff["radius"], axis=0).cpu()
    unif_r_diff = tf.concat(diff["unif_r"], axis=0).cpu()

    z_pred = tf.concat(dist["z_pred"], axis=0).cpu()
    z = tf.concat(dist["z"], axis=0).cpu()
    radius_pred = tf.concat(dist["radius_pred"], axis=0).cpu()
    radius = tf.concat(dist["radius"], axis=0).cpu()
    unif_radius_pred = tf.concat(dist["unif_r_pred"], axis=0).cpu()
    unif_radius = tf.concat(dist["unif_r"], axis=0).cpu()

    ## Plot histograms (diffs)
    plt.close()
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    fig.suptitle(f"Val. MSE: {total_val_loss:.2f} (MSE(z) + MSE(radius))\n\
    Avg. abs. diff. in z={z_pure_diff:.2f}, r={radius_pure_diff:.2f}")

    ## diff. plots
    z_diff_range = (-100, 100)
    axes[0,0].hist(z_diff, bins=20, range=z_diff_range, edgecolor='black')
    axes[0,0].set_title(r"z_diff ($z - \hat{z}$)")
    axes[0,0].set_xlabel('z difference')
    axes[0,0].set_ylabel('frequency')

    r_diff_range = (-100, 100)
    axes[0,1].hist(radius_diff, bins=20, range=r_diff_range, edgecolor='black')
    axes[0,1].set_title(r"radius_diff ($r - \hat{r}$)")
    axes[0,1].set_xlabel('radius diff')
    axes[0,1].set_ylabel('frequency')

    unif_r_diff_range = (-100, 100)
    axes[0,2].hist(unif_r_diff, bins=20, range=unif_r_diff_range, edgecolor='black')
    axes[0,2].set_title(r"(uniformalized) unif_radius ($r' - \hat{r'}$)")
    axes[0,2].set_xlabel('(uniformalized) radius diff')
    axes[0,2].set_ylabel('frequency')

    ## dist. plots
    # z_range = (min(z.min(), z_pred.min()).item(), max(z.max(), z_pred.max()).item())
    z_range = (-200, 200)
    axes[1,0].hist(z, bins=20, range=z_range, edgecolor='black', label="z")
    axes[1,0].hist(z_pred, bins=20, range=z_range, edgecolor='blue', label=r'$\hat{z}$', alpha=0.5)
    axes[1,0].set_title("z dist")
    axes[1,0].set_xlabel('z')
    axes[1,0].set_ylabel('frequency')

    # radius_range = (min(radius.min(), radius_pred.min()).item(), max(radius.max(), radius_pred.max()).item())
    radius_range = (0, 200)
    axes[1,1].hist(radius, bins=20, range=radius_range, edgecolor='black', label=r"$r$")
    axes[1,1].hist(radius_pred, bins=20, range=radius_range, edgecolor='blue', label=r"$\hat{r}$", alpha=0.5)
    axes[1,1].set_title("radius dist")
    axes[1,1].set_xlabel('radius')
    axes[1,1].set_ylabel('frequency')

    # unif_radius_range = (min(unif_radius.min(), unif_radius_pred.min()).item(), max(unif_radius.max(), unif_radius_pred.max()).item())
    unif_radius_range = (0, 150)
    axes[1,2].hist(unif_radius, bins=20, range=unif_radius_range, edgecolor='black', label=r"unif. $r$")
    axes[1,2].hist(unif_radius_pred, bins=20, range=unif_radius_range, edgecolor='blue', label=r"unif. $\hat{r}$", alpha=0.5)
    axes[1,2].set_title("(uniformalized) radius_dist")
    axes[1,2].set_xlabel(r'(uniformalized) radius $(r / 35) ^ {1 / 3}$')
    axes[1,2].set_ylabel('frequency')

    axes[1, 0].legend()
    axes[1, 1].legend()
    axes[1, 2].legend()

    plt.savefig(save_name + "_hist.png")
    plt.close()

def plot_reg(diff, dist, total_val_loss, abs_diff, save_name, args):
    if args.xyz_energy:
        abs_x_diff, abs_y_diff, abs_z_diff, abs_energy_diff = tf.reduce_mean(abs_diff, axis=0)
        energy_diff = tf.concat(diff["energy"], axis=0).cpu()
        energy_pred = tf.concat(dist["energy_pred"], axis=0).cpu()
        energy = tf.concat(dist["energy"], axis=0).cpu()
    else:
        abs_x_diff, abs_y_diff, abs_z_diff = tf.reduce_mean(abs_diff, axis=0)

    x_diff = tf.concat(diff["x"], axis=0).cpu()
    y_diff = tf.concat(diff["y"], axis=0).cpu()
    z_diff = tf.concat(diff["z"], axis=0).cpu()

    x_pred = tf.concat(dist["x_pred"], axis=0).cpu()
    y_pred = tf.concat(dist["y_pred"], axis=0).cpu()
    z_pred = tf.concat(dist["z_pred"], axis=0).cpu()

    x = tf.concat(dist["x"], axis=0).cpu()
    y = tf.concat(dist["y"], axis=0).cpu()
    z = tf.concat(dist["z"], axis=0).cpu()

    plt.close()
    if args.xyz_energy:
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
        fig.suptitle(f"Val. MSE: {total_val_loss:.2f} (MSE(x) + MSE(y) + MSE(y) + MSE(energy))\n\
        Avg. abs. diff. in x={abs_x_diff:.2f}, y={abs_y_diff:.2f}, z={abs_z_diff:.2f}, energy={abs_energy_diff:.2f}")
    else:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
        fig.suptitle(f"Val. MSE: {total_val_loss:.2f} (MSE(x) + MSE(y) + MSE(y))\n\
        Avg. abs. diff. in x={abs_x_diff:.2f}, y={abs_y_diff:.2f}, z={abs_z_diff:.2f}")

    ## diff. plots
    x_diff_range = (-50, 50)
    axes[0,0].hist(x_diff, bins=20, range=x_diff_range, edgecolor='black')
    axes[0,0].set_title(r"x_diff ($x - \hat{x}$)")
    axes[0,0].set_xlabel('x diff')
    axes[0,0].set_ylabel('freq')

    y_diff_range = (-50, 50)
    axes[0,1].hist(y_diff, bins=20, range=y_diff_range, edgecolor='black')
    axes[0,1].set_title(r"y_diff ($y - \hat{y}$)")
    axes[0,1].set_xlabel('y diff')
    axes[0,1].set_ylabel('freq')

    z_diff_range = (-50, 50)
    axes[0,2].hist(z_diff, bins=20, range=z_diff_range, edgecolor='black')
    axes[0,2].set_title(r"z_diff ($z - \hat{z}$)")
    axes[0,2].set_xlabel('z diff')
    axes[0,2].set_ylabel('freq')

    energy_diff_range = (0, 1)
    axes[0,3].hist(energy_diff, bins=20, range=energy_diff_range, edgecolor='black')
    axes[0,3].set_title(r"energy_diff ($energy - \hat{energy}$)")
    axes[0,3].set_xlabel('energy diff')
    axes[0,3].set_ylabel('freq')

    ## dist. plots
    x_range = (-250, 250)
    axes[1,0].hist(x, bins=20, range=x_range, edgecolor='black', label="x")
    axes[1,0].hist(x_pred, bins=20, range=x_range, edgecolor='blue', label=r'$\hat{x}$', alpha=0.5)
    axes[1,0].set_title("x dist")
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('freq')

    y_range = (-250, 250)
    axes[1,1].hist(y, bins=20, range=y_range, edgecolor='black', label="y")
    axes[1,1].hist(y_pred, bins=20, range=y_range, edgecolor='blue', label=r'$\hat{y}$', alpha=0.5)
    axes[1,1].set_title("y dist")
    axes[1,1].set_xlabel('y')
    axes[1,1].set_ylabel('freq')

    z_range = (-250, 250)
    axes[1,2].hist(x, bins=20, range=x_range, edgecolor='black', label="z")
    axes[1,2].hist(x_pred, bins=20, range=x_range, edgecolor='blue', label=r'$\hat{z}$', alpha=0.5)
    axes[1,2].set_title("z dist")
    axes[1,2].set_xlabel(r'z')
    axes[1,2].set_ylabel('freq')

    energy_range = (0, 4)
    axes[1,3].hist(energy, bins=20, range=energy_range, edgecolor='black', label="label")
    axes[1,3].hist(energy_pred, bins=20, range=energy_range, edgecolor='blue', label="pred", alpha=0.5)
    axes[1,3].set_title(r"energy_diff ($energy - \hat{energy}$)")
    axes[1,3].set_xlabel('energy diff')
    axes[1,3].set_ylabel('freq')

    axes[1, 0].legend()
    axes[1, 1].legend()
    axes[1, 2].legend()

    plt.savefig(save_name + "_hist.png")
    plt.close()
