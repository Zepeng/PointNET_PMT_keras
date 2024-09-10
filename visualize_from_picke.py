import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm, chisquare
import numpy as np

LOAD_NAME = './cgra/cgra_pointnet_8bit'

with open(f'{LOAD_NAME}.pickle', 'rb') as handle:
    save_data = pickle.load(handle)

# unpack from pickle file
abs_x_diff = save_data['abs_x_diff']
abs_y_diff = save_data['abs_y_diff']
abs_z_diff = save_data['abs_z_diff']
abs_energy_diff = save_data['abs_energy_diff']
energy_diff = save_data['energy_diff']
energy_pred = save_data['energy_pred']
energy = save_data['energy']
x_diff = save_data['x_diff']
y_diff = save_data['y_diff']
z_diff = save_data['z_diff']
x_pred = save_data['x_pred']
y_pred = save_data['y_pred']
z_pred = save_data['z_pred']
x = save_data['x']
y = save_data['y']
z = save_data['z']
total_val_loss = save_data['total_val_loss']

plt.close()
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 15))
# plt.subplots_adjust(wspace=0.2)
fig.suptitle(f"Validation Results\n Validation MSE: {total_val_loss:.2f}", fontsize=20)
#               (MSE(x) + MSE(y) + MSE(y) + MSE(energy))\n\
# Avg. abs. diff. in x={abs_x_diff:.2f}, y={abs_y_diff:.2f}, z={abs_z_diff:.2f}, energy={abs_energy_diff:.2f}", fontsize=20)
# else:
#     fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
#     fig.suptitle(f"Val. MSE: {total_val_loss:.2f} (MSE(x) + MSE(y) + MSE(y))\n\
#     Avg. abs. diff. in x={abs_x_diff:.2f}, y={abs_y_diff:.2f}, z={abs_z_diff:.2f}")


def plot_error_bar(axis, data, bins, name="unnamed", domain=100):
    # domain of 100 works well for cgra model
    filtered = data[(data>=-domain) & (data<=domain)]
    (mu, sigma) = norm.fit(filtered)
    pdf_fit = norm.pdf(bins, mu, sigma)
    # chi = chisquare(f_obs=pdf_fit, f_exp=data)
    print(f'{name} mu: {mu}, sigma: {sigma}, sigma (numpy): {np.std(data)}')
    axis.plot(bins, pdf_fit, 'r--', linewidth=5)

## diff. plots
x_diff_range = (-100, 100)
large_fontsize = 20
_, x_diff_bins, __ = axes[0,0].hist(x_diff, bins=20, range=x_diff_range, edgecolor='black', density=True)
axes[0,0].set_title(r"x_diff ($x - \hat{x}$)", fontsize=large_fontsize)
axes[0,0].set_xlabel('x diff', fontsize=large_fontsize)
axes[0,0].set_ylabel('Density', fontsize=large_fontsize)
plot_error_bar(axes[0,0], x_diff, x_diff_bins, name='x_diff')


y_diff_range = (-100, 100)
_, y_diff_bins, __ = axes[0,1].hist(y_diff, bins=20, range=y_diff_range, edgecolor='black', density=True)
axes[0,1].set_title(r"y_diff ($y - \hat{y}$)", fontsize=large_fontsize)
axes[0,1].set_xlabel('y diff', fontsize=large_fontsize)
plot_error_bar(axes[0,1], y_diff, y_diff_bins, name='y_diff')

z_diff_range = (-100, 100)
_, z_diff_bins, __ = axes[0,2].hist(z_diff, bins=20, range=z_diff_range, edgecolor='black', density=True)
axes[0,2].set_title(r"z_diff ($z - \hat{z}$)", fontsize=large_fontsize)
axes[0,2].set_xlabel('z diff', fontsize=large_fontsize)
plot_error_bar(axes[0,2], z_diff, z_diff_bins, name='z_diff')
# axes[0,2].set_ylabel('freq', fontsize=large_fontsize)

energy_diff_range = (-0.3, 0.3)
_, energy_diff_bins, __ = axes[0,3].hist(energy_diff, bins=20, range=energy_diff_range, edgecolor='black', density=True)
axes[0,3].set_title(r"energy_diff ($energy - \hat{energy}$)", fontsize=large_fontsize)
axes[0,3].set_xlabel('energy diff', fontsize=large_fontsize)
plot_error_bar(axes[0,3], energy_diff, energy_diff_bins, name='energy', domain=0.5)
# axes[0,3].set_ylabel('freq', fontsize=large_fontsize)

## dist. plots
x_range = (-250, 250)
axes[1,0].hist(x, bins=20, range=x_range, edgecolor='black', label="x")
axes[1,0].hist(x_pred, bins=20, range=x_range, edgecolor='blue', label=r'$\hat{x}$', alpha=0.5)
axes[1,0].set_title("x dist", fontsize=large_fontsize)
axes[1,0].set_xlabel('x (cm)', fontsize=large_fontsize)
axes[1,0].set_ylabel('Frequency', fontsize=large_fontsize)

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
axes[1,2].set_xlabel('z (cm)', fontsize=large_fontsize)
# axes[1,2].set_ylabel('freq', fontsize=large_fontsize)

energy_range = (-4, 4) #(0, 4)
axes[1,3].hist(energy, bins=20, range=energy_range, edgecolor='black', label="label")
axes[1,3].hist(energy_pred, bins=20, range=energy_range, edgecolor='blue', label="pred", alpha=0.5)
axes[1,3].set_title(r"energy_diff ($energy - \hat{energy}$)", fontsize=large_fontsize)
axes[1,3].set_xlabel('Energy diff (MeV)', fontsize=large_fontsize)
# axes[1,3].set_ylabel('freq', fontsize=large_fontsize)

axes[1, 0].legend(prop={'size':15})
axes[1, 1].legend(prop={'size':15})
axes[1, 2].legend(prop={'size':15})

plt.savefig(f'./{LOAD_NAME}_hist_with_fit.png')
plt.close()