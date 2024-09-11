import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('./data/pmt_xyz.dat', sep='\s+', header=None)
pmts = np.array(df)[:, 1:]
print(pmts)

fig = plt.figure(figsize=plt.figaspect(0.5))
model_ax = fig.add_subplot(1,2,1,projection='3d')
fpga_ax = fig.add_subplot(1,2,2,projection='3d')

model_ax.scatter(pmts.T[0], pmts.T[1], pmts.T[2], alpha=0.1)
fpga_ax.scatter(pmts.T[0], pmts.T[1], pmts.T[2], alpha=0.1)

fig.savefig('./animation.png')