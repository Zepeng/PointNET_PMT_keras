import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed
import sys
import os
import argparse
from tqdm import tqdm
import seaborn as sns

## .py imports ##
import sys
# from pointnet2_cls_msg import *
# from pointnet2_utils import *
sys.path.append('../')
from PointNet import * ## PointNetfeat
# from tools import * ## prepare_data_regression, train_graphs, add_positions, set_seed
from read_point_cloud import * ## get_pmtxyz
from utils import *
from time import time

clean_nohup_out()
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--use_wandb', action="store_true")
parser.add_argument('--reduce_lr_wait', type=int, default=20)
parser.add_argument('--enc_dropout', type=float, default=0.2)
parser.add_argument('--dec_dropout', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--smaller_run', action="store_true")
parser.add_argument('--dim_reduce_factor', type=float, default=1.5)
parser.add_argument('--batch_accum', type=int, default=1)
parser.add_argument('--no_Tnet', action="store_true")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--save_ver', default=0)
parser.add_argument('--trans_reg', type=float, default=0)
parser.add_argument('--feat_trans', action="store_true")
parser.add_argument('--res', action="store_true")
parser.add_argument('--max_only', action="store_true")
parser.add_argument('--mean_only', action="store_true")
parser.add_argument('--no_running_bn', action="store_true")
parser.add_argument('--add_dr_layer', action="store_true")
parser.add_argument('--radius_unif', action="store_true")
parser.add_argument('--seed', type=int, default=999)
parser.add_argument('--clf', action="store_true")
parser.add_argument('--joint_clf', action="store_true")
parser.add_argument('--momentum', type=float, default=0.1)
parser.add_argument('--no_bn', action="store_true")
parser.add_argument('--train_mode', action="store_true")
parser.add_argument('--warmup_bn', action="store_true")
parser.add_argument('--joint_clf_to_clf', action="store_true")
parser.add_argument('--xyz_label', action="store_true")
parser.add_argument('--xyz_energy', action="store_true")

parser.add_argument('--load_ver', help="Which version of model to load")
parser.add_argument('--debug_vis', action="store_true")
args = parser.parse_args()

accelerator = accelerate.Accelerator()
os.environ["WANDB_DISABLED"] = str(not args.use_wandb)
ddp_kwargs = accelerate.DistributedDataParallelKwargs(broadcast_buffers=False)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")


## file save name
suffix = "_no_Tnet" if args.no_Tnet else "_smaller_run" if args.smaller_run else ""
save_name = f"./{args.load_ver}/pointNET{suffix}"

## Load/Preprocess Data
with accelerator.main_process_first():
    pmtxyz = get_pmtxyz("data/pmt_xyz.dat", accelerator)
    accelerator.print("loading data...")
    ## Load data
    X, y = torch.load(f"./data/train_X_y_ver_all_xyz_energy.pt", map_location=torch.device("cpu"))
    if args.debug:
        accelerator.print("debug got called")
        small = 1000
        X, y = X[:small], y[:small]

    ## switch to match Aobo's syntax (time, charge, x, y, z) -> (x, y, z, label, time, charge)
    ## insert "label" feature to tensor. This feature (0 or 1) is the activation of sensor
    X, new_X = preprocess_features(X)

    ## Shuffle Data (w/ Seed)
    np.random.seed(seed=args.seed)
    set_seed(seed=args.seed)
    idx = np.random.permutation(new_X.shape[0]) 
    new_X, y = new_X[idx], y[idx]
    args.batch_size = int(args.batch_size // accelerator.num_processes) ##  Counter accelerator multiplying batch size by num_process

    ## Split and Load data
    train_split = 0.7
    val_split = 0.3
    train_idx = int(new_X.shape[0] * train_split)
    val_idx = int(train_idx + new_X.shape[0] * train_split)
    train = TensorDataset(new_X[:train_idx], y[:train_idx])
    val = TensorDataset(new_X[train_idx:val_idx], y[train_idx:val_idx])
    test = TensorDataset(new_X[val_idx:], y[val_idx:])
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    accelerator.print(f"num. total: {len(new_X)} train: {len(train)}, val: {len(val)}, test: {len(test)}")


## Init model
model = PointClassifier(
                n_hits=pmtxyz.shape[0], 
                dim=new_X.size(1), 
                out_dim=y.size(-1),
                dim_reduce_factor=args.dim_reduce_factor,
                args = args,
                )
nparam = sum([p.numel() for p in model.parameters()])
accelerator.print(f"num. parameters: {nparam}")
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
state_dict = torch.load(save_name+".pt", map_location=accelerator.device)
if hasattr(model, "module"):
    model.module.load_state_dict(state_dict)
else:
    model.load_state_dict(state_dict)

nparam = sum([p.numel() for p in model.parameters()])
accelerator.print(nparam)
model, train_loader, val_loader = accelerator.prepare(model, train_loader, val_loader)
args.model, args.nparam, args.n_device = model, nparam, accelerator.num_processes

epochs = range(args.epochs)
diff = {"x":[], "y":[], "z":[], "radius": [], "unif_r":[], "energy":[]}
dist = {"x":[], "y":[], "z":[], "x_pred":[], "y_pred":[], "z_pred":[], "energy":[], "energy_pred":[],
         "radius": [], "radius_pred": [], "unif_r": [], "unif_r_pred": []}
abs_diff = []
if args.train_mode:
    model.train()
else:
    model.eval()
with tqdm(total=len(val_loader), mininterval=5) as pbar, torch.no_grad():
    total_val_loss = 0

    accelerator.print("Validating...")
    for i, batch in enumerate(val_loader):
        X, y = batch
        out = model(X)
        out, y = accelerator.gather(out), accelerator.gather(y)
        abs_diff.append((y - out).abs())
        val_loss = F.mse_loss(out, y)
        total_val_loss += val_loss.item()

        diff_tensor = y - out ## to vis. distribution
        dist["x"].append(y[:, 0])
        dist["y"].append(y[:, 1])
        dist["z"].append(y[:, 2])

        dist["x_pred"].append(out[:, 0])
        dist["y_pred"].append(out[:, 1])
        dist["z_pred"].append(out[:, 2])
        
        diff["x"].append(diff_tensor[:, 0])
        diff["y"].append(diff_tensor[:, 1])
        diff["z"].append(diff_tensor[:, 2])

        dist["energy"].append(y[:, 3])
        dist["energy_pred"].append(out[:, 3])
        diff["energy"].append(diff_tensor[:, 3])

        pbar.update()
    total_val_loss /= len(val_loader)
    accelerator.print(total_val_loss)

dist, diff, abs_diff = accelerator.gather(dist), accelerator.gather(diff), accelerator.gather(abs_diff)
abs_diff = torch.cat(abs_diff)

## plot and save
plot_reg(diff=diff, dist=dist, total_val_loss=total_val_loss, abs_diff=abs_diff, save_name=save_name, args=args)