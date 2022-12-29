import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def data_load(data, node_num, input_len, output_len):
    scaler = StandardScaler()
    data_np = np.load(data)
    data_np = data_np['data'].astype('float32')
    print(data_np.shape)

    scaler.fit(data_np[:, 12-input_len:, 0])
    print(scaler)
    out_np = np.concatenate((scaler.transform(data_np[:, (12-input_len)*node_num:, 0]),data_np[:, :node_num*output_len, 1]), axis=1)
    print(out_np.shape)

    return out_np, scaler

