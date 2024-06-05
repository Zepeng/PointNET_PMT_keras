import torch
import numpy as np

X, y = torch.load(f"/sdf/group/exo/zpli/pointnet/train_X_y_ver_all_xyz_energy.pt", map_location=torch.device("cpu"))
X.to(torch.float32)
y.to(torch.float32)
X_np = X.numpy()
y_np = y.numpy()
outfile = np.savez('/sdf/group/exo/zpli/pointnet/train_X_y_ver_all_xyz_energy.npz', X=X_np, y = y_np)
