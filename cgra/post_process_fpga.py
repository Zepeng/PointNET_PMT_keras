import pandas as pd
import joblib
import numpy as np

df = pd.read_csv('./verified_output_from_fpga.csv')
target_scaler = joblib.load('target_scaler.gz')

# clean data and reogranize
print(df['output'])
reshaped = np.array(df['output']).reshape(32,4)/2**7 # divide since integer operationrs on fpga
print(reshaped)

fpga_output_scaled = target_scaler.inverse_transform(reshaped)
# now rescale
print(target_scaler.inverse_transform(reshaped))