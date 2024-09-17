import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import matplotlib.animation as animation
import functools

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

fig = plt.figure(figsize=plt.figaspect(0.5))
model_ax = fig.add_subplot(1,2,1,projection='3d')
fpga_ax = fig.add_subplot(1,2,2,projection='3d')
# energy_ax = fig.add_subplot(1,3,3)

# model_ax.scatter(pmts.T[0], pmts.T[1], pmts.T[2], alpha=0.1)
# fpga_ax.scatter(pmts.T[0], pmts.T[1], pmts.T[2], alpha=0.1)

artists = []
all_point_vals = np.reshape(np.array(X_vals), (64,2126,6))
for i in range(64):
    print(f'rendering frame {i}')
    # _, _, hist_container = energy_ax.hist(np.array(model_values).T[3][:i])
    # print(type(model_ax.scatter(model_values[i][0], model_values[i][1], model_values[i][2])))
    # so we may have to animate the histogram seperately since it uses a collection of artists
    # we have to draw the pmts from the x values. going to have (batch, 2126,1,6)
    point_vals = all_point_vals[i].T # rows are now in x,y,z,q,t, label
    # only want to show the pmts that were turned on
    point_vals = point_vals[:, np.where(point_vals[5, :] > 0)[0]] # may need to double check this line, but works for now
    to_append = [
            model_ax.scatter(point_vals[0], point_vals[1], point_vals[2], color='red', alpha=0.05),
            model_ax.scatter(model_values[i][0], model_values[i][1], model_values[i][2], label='Predicted', color='green'),
            model_ax.scatter(target_values[i][0], target_values[i][1], target_values[i][2], label='True', color='orange'),
            fpga_ax.scatter(pmts.T[0], pmts.T[1], pmts.T[2], alpha=0.1, color='green'),
    ]
    # if i==0:
    #     to_append.append(model_ax.legend())
    artists.append(to_append)

# adapted from https://matplotlib.org/stable/gallery/animation/animated_histogram.html
def animate(frame_number, bar_container):
    # print(model_energies[:frame_number+1])
    n, _ = np.histogram(model_energies[:frame_number+1], np.linspace(-4,4,100))
    for count, rect in zip(n, bar_container.patches):
        rect.set_height(count)
    return bar_container.patches


spatial_ani = animation.ArtistAnimation(fig=fig, artists=artists)
energy_fig, energy_ax = plt.subplots()
_, _, bar_container = energy_ax.hist(np.random.rand(100), np.linspace(-4,4,100))
energy_ax.set_ylabel('Count')
energy_ax.set_xlabel('Energy (MeV)')
energy_anim = functools.partial(animate, bar_container=bar_container)
energy_ani = animation.FuncAnimation(energy_fig, energy_anim, 64, repeat=True, blit=True)

writer = animation.PillowWriter(fps=2)

spatial_ani.save('scatter.gif', writer=writer)
energy_ani.save('energy.gif', writer=writer)

fig.savefig('./animation.png')