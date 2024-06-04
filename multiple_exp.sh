#!/bin/bash


# ## Train Model
nohup accelerate launch train.py --use_wandb --epochs 600 --enc_dropout 0 --dec_dropout 0.1 --weight_decay 1e-3 --lr 1e-3 --save_ver 13 \
                                --dim_reduce_factor 2 --batch_size 256 --mean_only --save_best --patience 15 --energy_mult 10 ##--debug

## Visualize
nohup accelerate launch vis_for_pointNET.py --dim_reduce_factor 2 --batch_size 256 --mean_only --seed 999 \
                                            --load_ver 13 --xyz_energy