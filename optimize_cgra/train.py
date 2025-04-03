from tqdm import tqdm
import tensorflow as tf
from keras.models import Model, save_model
import time
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib
import pytest
from tensorflow import keras
from keras.optimizers import Adam
import json
from optimize_cgra.util import *
import time

def traina(model, train_loader, target_scaler, val_loader):
    B_EPOCH = 2
    BATCH_SIZE = 64
    VALIDATION_SPLIT = 0.1
    TRAINING_EPOCHS = 2
    DEBUG = False
    training = True

    print(f"Model is now training for {TRAINING_EPOCHS} epochs")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    epochs = range(TRAINING_EPOCHS)

    
    pbar = tqdm(total=TRAINING_EPOCHS, mininterval=10)
    # Initialize best validation loss and best train loss
    best_val, best_train = float("inf"), float("inf")
    tot_train_lst = []
    epoch_losses = {}

    for epoch in range(TRAINING_EPOCHS):
        total_loss = 0
        ts = time.time()

        for i, (X, y) in enumerate(train_loader):
            
            # # ADD COMMENTS ON WHY DOING THIS
            try:
                X = tf.convert_to_tensor(X.numpy().reshape((BATCH_SIZE, 2126, 1, 6)))
            except ValueError:
                print("skipping batch due to incompatible size")
                break

            index = 0
            with tf.GradientTape() as tape:
                out = model(X)
                energy_mult = 160

                loss = tf.reduce_mean(tf.keras.losses.MSE(out, y))
                if index % 100 == 0:
                    out = tf.convert_to_tensor(target_scaler.inverse_transform(out.numpy()))
                    y = tf.convert_to_tensor(target_scaler.inverse_transform(y.numpy()))
        
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            total_loss += loss.numpy()


        total_loss /= len(train_loader)
        # wandb.log({"Loss":total_loss})
        tot_train_lst.append(total_loss)
        pbar.update(1)
        # print(total_loss)

        epoch_losses[f"epoch_{epoch + 1}"] = total_loss
        print(f"Finish epoch {epoch}, total train loss {total_loss}, time elapsed {time.time() - ts}")
        tf.keras.backend.clear_session()

    with open("training_loss.json", "w") as f:
        json.dump(epoch_losses, f, indent=4)
    
    print("Training complete. Losses saved to training_loss.json")

    pbar.close()

    min_train = round(min(tot_train_lst), 2)
    min_values = {
            "min_train": min_train,
            "min_val": best_val,
    }

# def eval(model, val_loader, BATCH_SIZE, target_scaler):

#     diff = {"x":[], "y":[], "z":[], "radius": [], "unif_r":[], "energy":[]}
#     dist = {"x":[], "y":[], "z":[], "x_pred":[], "y_pred":[], "z_pred":[], "energy":[], "energy_pred":[],
#             "radius": [], "radius_pred": [], "unif_r": [], "unif_r_pred": []}
#     abs_diff = []

#     print('Model eval')
#     scale_factor = 1#25.
#     collect_export_values = True

#     with tqdm(total=len(val_loader), mininterval=5) as pbar:
#         total_val_loss = 0

#         for i, batch in enumerate(val_loader):
#             X, y = batch
#             print(f"Collecting values for hardware accuracy test: {collect_export_values}")
#             try:
#                 X = tf.convert_to_tensor(X.numpy().reshape((BATCH_SIZE, 2126, 1, 6)))
#                 if collect_export_values:
#                     global deploy_val_X
#                     deploy_val_X = X

#                     with open(f'X.pickle', 'wb') as handle:
#                         pickle.dump(X.numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)
#             except ValueError:
#                 print("skipping batch due to incompatible size")
#                 break
#             out = model(X)
#             out = tf.convert_to_tensor(target_scaler.inverse_transform(out))
#             y = tf.convert_to_tensor(target_scaler.inverse_transform(y))
#             if collect_export_values:
#                 global deploy_val_model
#                 global deploy_val_Y_truth

#                 deploy_val_model = out
#                 deploy_val_Y_truth = y
#                 collect_export_values = False

#                 with open("accuracy_test.json", 'w') as fp:
#                     json.dump({
#                         'model_value': out.numpy().tolist(),
#                         'target_value': y.numpy().tolist(),
#                         'X_vals': deploy_val_X.numpy().tolist()
#                     }, fp)

#             abs_diff.append(tf.abs(y*scale_factor - out*scale_factor))
#             val_loss = tf.reduce_mean(tf.keras.losses.MSE(out, y))
#             total_val_loss += val_loss.numpy()

#             diff_tensor = (y - out)*scale_factor ## to vis. distribution
#             dist["x"].append(y[:, 0]*scale_factor)
#             dist["y"].append(y[:, 1]*scale_factor)
#             dist["z"].append(y[:, 2]*scale_factor)

#             dist["x_pred"].append(out[:, 0]*scale_factor)
#             dist["y_pred"].append(out[:, 1]*scale_factor)
#             dist["z_pred"].append(out[:, 2]*scale_factor)
            
#             diff["x"].append(diff_tensor[:, 0])
#             diff["y"].append(diff_tensor[:, 1])
#             diff["z"].append(diff_tensor[:, 2])

#             dist["energy"].append(y[:, 3]*scale_factor)
#             dist["energy_pred"].append(out[:, 3]*scale_factor)
#             diff["energy"].append(diff_tensor[:, 3])

#             pbar.update()
#             # print(f"\nValidation Loss at epoch: {i} is {np.mean(losses)}")
#         total_val_loss /= len(val_loader)
#         tf.keras.backend.clear_session()

#     abs_diff = tf.concat(abs_diff, axis=0)
#     plot_and_save(model, abs_diff, diff, dist, total_val_loss)

    diff = {"x":[], "y":[], "z":[], "radius": [], "unif_r":[], "energy":[]}
    dist = {"x":[], "y":[], "z":[], "x_pred":[], "y_pred":[], "z_pred":[], "energy":[], "energy_pred":[],
                "radius": [], "radius_pred": [], "unif_r": [], "unif_r_pred": []}
    abs_diff = []

    print('Model eval')
    scale_factor = 1#25.
    collect_export_values = True

    with tqdm(total=len(val_loader), mininterval=5) as pbar:
        total_val_loss = 0

        for i, batch in enumerate(val_loader):
            X, y = batch
            print(f"Collecting values for hardware accuracy test: {collect_export_values}")
            try:
                X = tf.convert_to_tensor(X.numpy().reshape((BATCH_SIZE, 2126, 1, 6)))
                if collect_export_values:
                    global deploy_val_X
                    deploy_val_X = X

                    with open(f'X.pickle', 'wb') as handle:
                        pickle.dump(X.numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)
            except ValueError:
                print("skipping batch due to incompatible size")
                break
            out = model(X)
            # do inverse transform on data
            # new_shape = out.shape
            # print(X.shape)
            # print(out.shape)
            # print(y.shape)
            out = tf.convert_to_tensor(target_scaler.inverse_transform(out))
            y = tf.convert_to_tensor(target_scaler.inverse_transform(y))
            if collect_export_values:
                global deploy_val_model
                global deploy_val_Y_truth

                deploy_val_model = out
                deploy_val_Y_truth = y
                collect_export_values = False
                # now save these values
                with open("accuracy_test.json", 'w') as fp:
                    json.dump({
                        'model_value': out.numpy().tolist(),
                        'target_value': y.numpy().tolist(),
                        'X_vals': deploy_val_X.numpy().tolist()
                    }, fp)

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
        tf.keras.backend.clear_session()

    abs_diff = tf.concat(abs_diff, axis=0)

    ## plot and save

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

    save_model(model, 'cgra_model.h5')

    
    






    






        


         

    