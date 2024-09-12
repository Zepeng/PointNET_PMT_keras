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
    model_values = acc_test['model_value']
    target_values = acc_test['target_value']

model_energies = np.array(model_values).T[3]
target_energies = np.array(target_values).T[3]

fig = plt.figure(figsize=plt.figaspect(0.5))
model_ax = fig.add_subplot(1,2,1,projection='3d')
fpga_ax = fig.add_subplot(1,2,2,projection='3d')
# energy_ax = fig.add_subplot(1,3,3)

model_ax.scatter(pmts.T[0], pmts.T[1], pmts.T[2], alpha=0.1)
# fpga_ax.scatter(pmts.T[0], pmts.T[1], pmts.T[2], alpha=0.1)

artists = []
for i in range(5):
    # _, _, hist_container = energy_ax.hist(np.array(model_values).T[3][:i])
    # print(type(model_ax.scatter(model_values[i][0], model_values[i][1], model_values[i][2])))
    # so we may have to animate the histogram seperately since it uses a collection of artists
    artists.append(
        [
            model_ax.scatter(model_values[i][0], model_values[i][1], model_values[i][2], label='Predicted', color='green'),
            model_ax.scatter(target_values[i][0], target_values[i][1], target_values[i][2], label='True', color='orange'),
            fpga_ax.scatter(pmts.T[0], pmts.T[1], pmts.T[2], alpha=0.1, color='green'),
    ])


def animate(frame_number, bar_container):
    n, _ = np.histogram(model_energies[:frame_number+1], np.linspace(-4,4,100))
    for count, rect in zip(n, bar_container.patches):
        rect.set_height(count)
    return bar_container.patches


spatial_ani = animation.ArtistAnimation(fig=fig, artists=artists)
energy_fig, energy_ax = plt.subplots()
_, _, bar_container = energy_ax.hist(np.random.rand(100), np.linspace(-4,4,100))
energy_anim = functools.partial(animate, bar_container=bar_container)
energy_ani = animation.FuncAnimation(energy_fig, energy_anim, 50, repeat=True, blit=True)

writer = animation.PillowWriter(fps=2)

spatial_ani.save('scatter.gif', writer=writer)
energy_ani.save('energy.gif', writer=writer)

fig.savefig('./animation.png')