import numpy as np
import torch
import pandas as pd

## load pmt pos into a tensor
def get_pmtxyz(fname, accelerator):
    df = pd.read_csv(fname, sep='\s+', header=None)
    pmtpos = torch.tensor(df.values)
    pmtxyz = pmtpos[:, 1:] # get rid of redundant id column
    accelerator.print(f"pmtpos shape: {pmtxyz.shape}") # [2126, 3]
    return pmtxyz