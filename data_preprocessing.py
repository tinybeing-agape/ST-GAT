import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def data_load_n_window(data):
    data_np = np.load('./data/' + data)
    data_np = data_np['data']
    # data_np = np.swapaxes(data_np, 0, 1)
    print(data_np.shape)
    # scaler.fit(data_np)
    # data_np = scaler.transform(data_np)
    data_n = np.array([])
    for i in range(0, len(data_np)-12):
        if i > 0:
            data_n = np.append(data_n, data_np[i:i+12], axis=0)
        else:
            data_n = data_np[i:i+12]
        if i % 500 == 0:
            print(i)
    data_n = data_n.reshape(-1, data_np.shape[1] * 12, 1)
    data_x = data_n[:-12]
    data_y = data_n[12:]

    print(data_x.shape, data_y.shape)
    data_np = np.append(data_x, data_y, axis=2)

    return data_np


def data_load_n_window_h5(data):
    h5 = pd.HDFStore('C:/Users/joker/Desktop/newidea/data/' + data, 'r')
    print(h5.keys())
    h5 = h5['/df']
    h5 = h5.to_numpy()
    print(h5.shape)
    data_n = np.array([])
    for i in range(0, len(h5)-12):
        if i > 0:
            data_n = np.append(data_n, h5[i:i+12], axis=0)
        else:
            data_n = h5[i:i+12]
    data_n = data_n.reshape(-1, h5.shape[1] * 12, 1)
    data_x = data_n[:-1]
    data_y = data_n[1:]

    print(data_x.shape, data_y.shape)
    data_np = np.append(data_x, data_y, axis=2)

    return data_np

def data_load(data, node_num):
    scaler = StandardScaler()
    data_np = np.load(data)
    data_np = data_np['data'].astype('float32')
    print(data_np.shape)
    # data_np = data_np.reshape(-1, node_num, 2)
    scaler.fit(data_np[:, :, 0])
    scaler.partial_fit(data_np[:, :, 1])
    print(scaler)
    data_np[:, :, 0] = scaler.transform(data_np[:, :, 0])
    data_np[:, :, 1] = scaler.transform(data_np[:, :, 1])
    print(data_np.shape)
    # data_np = data_np.reshape(-1, node_num*12, 2)

    return data_np, scaler


if __name__ == '__main__':
    # h5 = pd.HDFStore('C:/Users/joker/Desktop/newidea/data/metr-la.h5', 'r')
    # print(h5.keys())
    # h5 = h5['/df']
    # h5 = h5.to_numpy()
    # print(h5.shape)
    # np.savez('C:/Users/joker/Desktop/newidea/data/metr-la', data=h5)
    # topy = data_load_n_window_h5('metr-la.h5')
    # np.savez('C:/Users/joker/Desktop/newidea/data/r_window_metr-la', data=topy)
    # topy, _ = data_load_n_window('HY.npz')
    # np.savez('C:/Users/joker/Desktop/newidea/data/r_window_HY', data=topy)
    # # print(data_load('pems04.npz')[0])
    # topy = data_load_n_window('metr-la.npz')
    # np.savez('C:/Users/joker/Desktop/newidea/data/window_metr-la_', data=topy)
    dt = np.load('./data/window_metr-la.npz')['data']
    print(dt)
    # data_np = np.load('C:/Users/joker/Desktop/newidea/data/r_window_seoul_high.npz')
    # data_np = data_np['data']
    # tmp_x = data_np[:-11, :, 0]
    # tmp_y = data_np[11:, :, 1]
    # tmp_x = tmp_x[:, :, np.newaxis]
    # tmp_y = tmp_y[:, :, np.newaxis]
    # tmp = np.append(tmp_x, tmp_y, axis=2)
    # print(tmp.shape)
    # np.savez('C:/Users/joker/Desktop/newidea/data/window_seoul_high', data=tmp)
    # dt = np.load('./data/metr-la.npz')['data']
    # print(dt.shape)
    # np.savetxt('metr-la.csv', dt, fmt='%.18e', delimiter=',')
