import numpy as np
import tensorflow as tf 
import pandas as pd

## load pmt pos into a tensor
def get_pmtxyz(fname):
    df = pd.read_csv(fname, sep='\s+', header=None)
    pmtpos = tf.convert_to_tensor(df.values)
    pmtxyz = pmtpos[:, 1:] # get rid of redundant id column
    print(f"pmtpos shape: {pmtxyz.shape}") # [2126, 3]
    return pmtxyz
