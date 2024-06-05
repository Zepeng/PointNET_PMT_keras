import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import wandb
import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed
import sys
import os
import argparse
from tqdm import tqdm
from pprint import pprint
from time import time

## .py imports ##
from PointNet import *
from read_point_cloud import * 
from utils import *

strt = time()
clean_nohup_out()
parser = argparse.ArgumentParser()
## Hyperparameters
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--use_wandb', action="store_true")
parser.add_argument('--reduce_lr_wait', type=int, default=20)
parser.add_argument('--enc_dropout', type=float, default=0.2)
parser.add_argument('--dec_dropout', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--smaller_run', action="store_true")
parser.add_argument('--dim_reduce_factor', type=float, default=1.5)
parser.add_argument('--debug', action="store_true")
parser.add_argument('--save_ver', default=0)
parser.add_argument('--mean_only', action="store_true")
parser.add_argument('--save_best', action="store_true")
parser.add_argument('--seed', type=int, default=999)
parser.add_argument('--patience', type=int, default=15)


## Initiate accelerator (distributed training) + flush output to {ver}/train.txt
args = parser.parse_args()
ver = args.save_ver
accelerator = accelerate.Accelerator()
os.environ["WANDB_DISABLED"] = str(not args.use_wandb or args.debug)
ddp_kwargs = accelerate.DistributedDataParallelKwargs()
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")
accelerator.print(f"ver: {ver}")
init_logfile(ver)

## file save name
save_name = f"./{ver}/pointNET"


## Load/Preprocess Data
with accelerator.main_process_first():
    accelerator.print("loading data...")
    ## Load data
    pmtxyz = get_pmtxyz("data/pmt_xyz.dat", accelerator=accelerator)
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
    print(f"num. total: {len(new_X)} train: {len(train)}, val: {len(val)}, test: {len(test)}")


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


## Optimizers, Scheduler, Prepare Distributed Training
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
## always min mode scheduler because we are using train loss as ref.
scheduler = ReduceLROnPlateau(optimizer, "min", 0.80, patience=args.patience*accelerator.num_processes, min_lr=1e-5, threshold=1e-3, threshold_mode='rel') 


## accelerator.prepare - distributed training
model, train_loader, val_loader, optimizer, scheduler = accelerator.prepare(model, train_loader, val_loader, optimizer, scheduler)
args.model, args.nparam, args.n_device = model, nparam, accelerator.num_processes
args.batch_size *= accelerator.num_processes ## bring the batchsize back up ONLY for logging purposes
accelerator.init_trackers("pointNET", config=vars(args))
if accelerator.is_main_process:
    print("\n\nHyperparameters being used:\n")
    pprint(vars(args))
    print("\n\n")
epochs = range(args.epochs)


## Train Loop
with tqdm(total=args.epochs, mininterval=10) as pbar:
    best_val, best_train = float("inf"), float("inf")
    tot_train_lst = []
    for epoch in epochs:
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            ## forward prop
            X, y = batch
            out = model(X)
            energy_mult = 160 ## scale energy loss
            out[:, -1], y[:, -1] = energy_mult * out[:, -1], energy_mult * y[:, -1] 
            loss = F.mse_loss(out, y)
            train_loss = loss 

            ## backward prop
            accelerator.backward(train_loss)

            ## logging purposes
            with torch.no_grad():
                total_loss += train_loss.item()
            ## update grad
            optimizer.step()
            optimizer.zero_grad()

        total_loss /= len(train_loader)
        tot_train_lst.append(total_loss)

        ## reduce lr on pleatau
        accelerator.print(f"min train loss: {min(tot_train_lst)}")
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(total_loss)

        ## validation
        if epoch % 10 == 0 or epoch == args.epochs-1:
            with torch.no_grad():
                total_val_loss = 0
                ## validation (eval mode)
                model.eval()
                for batch in val_loader:
                    ## forward
                    X, y = batch
                    out = model(X)
                    out = accelerator.gather(out)
                    y = accelerator.gather(y)
                    ## compute metric
                    val_loss = F.mse_loss(out, y)
                    total_val_loss += val_loss.item()
                total_val_loss /= len(val_loader)
            current_lr = optimizer.param_groups[0]['lr']
            accelerator.print(f'\nepoch {epoch}, train loss: {total_loss:.2f}, val loss: {total_val_loss:.2f}, lr: {current_lr}')
            accelerator.log({
                "train loss": total_loss,
                "val_loss": total_val_loss,
                "lr": current_lr,
            })

            if accelerator.is_main_process:
                pbar.update(10) ## tqdm: only show update once

            ## save best (val loss) model
            cond = total_val_loss < best_val
            if cond:
                best_val = total_val_loss
                if args.save_best:
                    if hasattr(model, "module"):
                        torch.save(model.module.state_dict(), save_name+".pt")
                    else:
                        torch.save(model.state_dict(), save_name+".pt")
                    accelerator.print(f"New Best Score!! Model_saved as {save_name}")
        else:
            accelerator.log({"train loss": total_loss})

## time
tot_time = time() - strt
accelerator.log({"time": tot_time})
accelerator.print(f"total time taken: {tot_time}")

## min values
min_train = round(min(tot_train_lst), 2)
min_values = {
        "min_train": min_train,
        "min_val": best_val,
}
if accelerator.is_main_process:
    pprint(min_values)
accelerator.log(min_values)