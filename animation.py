import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import matplotlib.animation as animation
import functools
import joblib

FRAMES = 64
fpga_batch_size = 64
fpga_latency = 34099.11330 # from vitis, in ms
fps_real = 1/(fpga_latency/fpga_batch_size/1000)
print(fps_real)
assert fps_real > 1 # should be at least 1 fps from what we've seen so far

# load the pmt locations
df = pd.read_csv('./data/pmt_xyz.dat', sep='\s+', header=None)
pmts = np.array(df)[:, 1:]

# load the accuracy test from import
with open('./cgra/accuracy_test.json', 'r') as file:
    acc_test = json.load(file)
    global model_values
    global target_values
    global X_vals
    model_values = acc_test['model_value']
    target_values = acc_test['target_value']
    X_vals = acc_test['X_vals']

model_energies = np.array(model_values).T[3]
target_energies = np.array(target_values).T[3]

# # now load the fpga output
# with open('./cgra/fpga output.json', 'r') as fp:
#     global fpga_vals
#     fpga_vals = json.load(fp)['vals']

# load fpga outpu
# import numpy as np

df = pd.read_csv('./cgra/64_batch_output.csv')
target_scaler = joblib.load('./cgra/target_scaler.gz')

# clean data and reogranize
print(df['output'])
reshaped = np.array(df['output']).reshape(fpga_batch_size,4)/2**7 # divide since integer operationrs on fpga
# print(reshaped)
fpga_output_scaled = target_scaler.inverse_transform(reshaped)
fpga_energies = fpga_output_scaled.T[3]
# print(fpga_energies)
# assert 0

fig = plt.figure()
model_ax = fig.add_subplot(1,2,1,projection='3d')
fpga_ax = fig.add_subplot(1,2,2,projection='3d')
fig.tight_layout(pad=0.15)

model_ax.set_axis_off()
fpga_ax.set_axis_off()
# energy_ax = fig.add_subplot(1,3,3)

# model_ax.scatter(pmts.T[0], pmts.T[1], pmts.T[2], alpha=0.1)
# fpga_ax.scatter(pmts.T[0], pmts.T[1], pmts.T[2], alpha=0.1)

artists = []
all_point_vals = np.reshape(np.array(X_vals), (64,2126,6))
for i in range(FRAMES):
    print(f'Rendering vertex frame {i+1}')
    # _, _, hist_container = energy_ax.hist(np.array(model_values).T[3][:i])
    # print(type(model_ax.scatter(model_values[i][0], model_values[i][1], model_values[i][2])))
    # so we may have to animate the histogram seperately since it uses a collection of artists
    # we have to draw the pmts from the x values. going to have (batch, 2126,1,6)
    point_vals = all_point_vals[i].T # rows are now in x,y,z,q,t, label
    # only want to show the pmts that were turned on
    point_vals = point_vals[:, np.where(point_vals[5, :] > 0)[0]] # may need to double check this line, but works for now
    to_append = [
            model_ax.scatter(point_vals[0], point_vals[1], point_vals[2], color='red', alpha=0.05),
            # model_ax.scatter(model_values[i][0], model_values[i][1], model_values[i][2], label='Predicted', color='green'),
            # model_ax.scatter(target_values[i][0], target_values[i][1], target_values[i][2], label='True', color='orange'),
            fpga_ax.scatter(pmts.T[0], pmts.T[1], pmts.T[2], alpha=0.1, color='blue'),
            fpga_ax.scatter(fpga_output_scaled[i][0], fpga_output_scaled[i][1], fpga_output_scaled[i][2], color='red'),
            fpga_ax.scatter(target_values[i][0], target_values[i][1], target_values[i][2], label='True', color='orange')
    ]
    # if i==0:
    #     to_append.append(model_ax.legend())
    artists.append(to_append)

# adapted from https://matplotlib.org/stable/gallery/animation/animated_histogram.html
def animate(frame_number, bar_container):
    # print(model_energies[:frame_number+1])
    n, _ = np.histogram(fpga_energies[:frame_number+1], np.linspace(-4,4,100))
    for count, rect in zip(n, bar_container.patches):
        rect.set_height(count)
    return bar_container.patches


spatial_ani = animation.ArtistAnimation(fig=fig, artists=artists)

energy_fig, energy_ax = plt.subplots()
_, _, bar_container = energy_ax.hist(target_energies[:FRAMES], np.linspace(0,4,15), color='purple', histtype='step')
energy_ax.set_ylabel('Count')
energy_ax.set_xlabel('Energy (MeV)')

# try animating using artist animation now
bar_artists = []
for frame in range(FRAMES):
    print(f'Rendering energy frame {frame+1}')
    _, _, patches = energy_ax.hist(fpga_energies[:frame+1], np.linspace(0,4,15), color='green')
    bar_artists.append(patches)

# energy_anim = functools.partial(animate, bar_container=bar_container)
# energy_ani = animation.FuncAnimation(energy_fig, energy_anim, 32, repeat=True, blit=True)
energy_ani = animation.ArtistAnimation(fig=energy_fig, artists=bar_artists)

writer = animation.PillowWriter(fps=fps_real)

spatial_ani.save('scatter.gif', writer=writer)
energy_ani.save('energy.gif', writer=writer)

fig.savefig('./animation.png')