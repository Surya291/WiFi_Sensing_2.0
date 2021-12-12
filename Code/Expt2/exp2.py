# %%
from sklearn.preprocessing import StandardScaler
import math
import sys
import json
import argparse
import pandas as pd
import numpy as np
import scipy.signal as signal
from sklearn import preprocessing
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.express as px
from hampel import hampel
import h5py

# %%
PATH = '/home/kuntal990/projects/WiFi_Sensing_2.0/ml/exp04/page/page1.csv'
# %%


def plot_rssi(PATH=PATH):
    df = pd.read_csv(
        PATH)
    df_rssi = df.loc[:, ['rssi']]
    df_time = df.loc[:, ['real_timestamp']]
    # print(df_rssi)
    # print(df_time)
    df_rssi.plot(y=['rssi'])
    plt.axis([0, len(df_rssi.index), -72, -45])
    #plt.plot(df_time['real_timestamp'], df_rssi['rssi'])
    plt.ylabel('rssi magnitude')


def load_array(PATH=PATH):
    df = pd.read_csv(
        PATH)
    df_csi = df.loc[:, ['len', 'CSI_DATA']]
    drop_idx = []
    for i in range(df_csi.shape[0]):
        if df_csi.iloc[i]['len'] < 384:
            drop_idx.append(i)

    df_csi = df_csi.drop(drop_idx)
    size_x = len(df_csi.index)
    size_y = df_csi.iloc[0]['len']//2  # no. of subcarriers ..

    array_csi = np.zeros([size_x, size_y], dtype=np.complex64)

    for x, csi in enumerate(df_csi.iloc):
        temp = csi["CSI_DATA"].replace(' ', ',')
        temp = temp.replace(',]', ']')
        # print(temp)
        ll = 0
        if temp[-1] != ']':
            temp = temp + ']'
            ll = -1
        temp = temp.replace(',]', ']')
        temp = temp.replace('-]', ']')
        # print((temp))
        temp = temp.replace(',]', ']')
        csi_raw_data = json.loads(temp)
        ll += len(csi_raw_data)
        for y in range(0, ll, 2):
            # IQ channel frequency response
            array_csi[x][y//2] = complex(csi_raw_data[y], csi_raw_data[y + 1])
    return array_csi


def running_mean(x, N):
    cumsum = np.cumsum(x, axis=1)
    tmp = np.zeros(shape=cumsum.shape)
    for i in range(len(x)):
        tmp[i] = tmp[i] + (cumsum[i] - cumsum[max(i-N, 0)]) / float(N)

        tmp[i] = tmp[i] + (cumsum[min(i+N, len(x)-1)] - cumsum[i]) / float(N)
    return tmp


def remove_offset(x, w=200):
    #cumsum = np.cumsum(x, axis=1)
    for i in range(0, len(x)-1, w):
        # print(i)
        start = i
        end = min(i + w, len(x)-1)
        #offset = (cumsum[end] - cumsum[start])/float(end - start)
        offset = np.mean(x[start:end])
        x[start:end] = x[start:end] - offset
    return x


def hampel_smooth(time_series, axis=1):
    ham_series = time_series.copy()
    for sc in range(time_series.shape[axis]):
        # print(sc)
        sc_ts = pd.Series(time_series[:, sc])
        # print(sc_ts.shape)
        ham_series[:, sc] = hampel(sc_ts, window_size=20, n=3, imputation=True)

    return ham_series


def hampel_smooth2(vals_orig, k=7, t0=3):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''

    #Make copy so original not edited
    vals = vals_orig.copy()
    vals = pd.DataFrame(vals)
    #Hampel Filter
    L = 1.4826
    rolling_median = vals.rolling(window=k, center=True, axis = 0).median()
    def MAD(x): return np.median(np.abs(x - np.median(x, axis=0)))
    rolling_MAD = vals.rolling(window=k, center=True, axis=0).apply(MAD)
    threshold = t0 * L * rolling_MAD
    difference = np.abs(vals - rolling_median)

    '''
    Perhaps a condition should be added here in the case that the threshold value
    is 0.0; maybe do not mark as outlier. MAD may be 0.0 without the original values
    being equal. See differences between MAD vs SDV.
    '''

    outlier_idx = difference > threshold
    vals[outlier_idx] = rolling_median[outlier_idx]
    return(vals)


#%%
BS = 250
dataset = []
labels = []


dataset2 = []
labels2 = []

# %%
#PATH = '/home/kuntal990/projects/WiFi_Sensing_2.0/ml/exp04/page/page2.csv'
# plot_rssi(PATH)
#data = abs(load_array(PATH))
plt.plot(data[:, 6][:1200])
plt.title('Without SMA')
plt.show()

data = hampel_smooth(data)
plt.plot(data[:, 6][:1200])
plt.title('With Hampel')
plt.show()

data = remove_offset(data, 300)
plt.plot(data[:, 6][:1200])
plt.title('With hampel and offset')
#plt.ylim(-0.5, 0.5)
plt.show()
#%%
PATH = '/home/kuntal990/projects/WiFi_Sensing_2.0/ml/exp04/phone'
data = abs(load_array(PATH + '/phone' +str(1) + '.csv'))[:250]

for i in range(2, 31):
    tmp = abs(load_array(PATH + '/phone' +str(i) + '.csv'))
    if len(tmp) < 250:
        continue
    data = np.concatenate((data, tmp[:250]), axis = 0)
    print(len(data))
    # plt.plot(data[:, 6][:1200])
    # plt.title('Without SMA')
    # plt.show()



# %%
#create dataset

data = StandardScaler().fit_transform(data)

steps = data.shape[0]//BS

for i in range(steps-1):
    batch = data[i*BS: min((i+1)*BS, data.shape[0]),:]
    dataset.append(batch)
    labels.append(4)

#%%
dataset = np.array(dataset)
labels = np.array(labels)
labels = labels.reshape(-1, 1)
#%%
dataset3 = np.concatenate((dataset2, dataset), axis=0)


#%%

with h5py.File('/home/kuntal990/projects/WiFi_Sensing_2.0/dataset/dataset4_final.hdf5', 'w') as hf:
    X = hf.create_dataset('X', data=dataset)
    Y = hf.create_dataset('Y', data=labels)

hf.close()
# %%

X_train2 = []
y_train2 = []

with h5py.File('/home/kuntal990/projects/WiFi_Sensing_2.0/dataset/dataset2_final.hdf5', 'r') as hf:
    X = hf.get('X')
    X_train2 = np.array(X)
    Y = hf.get('Y')
    y_train2 = np.array(Y)
hf.close()

#%%  
chunked_data = np.array_split(data, 12)
for i, xx in enumerate(chunked_data):
    plt.title(f'chunk {i+1}')
    plt.plot(xx[:, 6])
    plt.show()

# %%
pca = PCA(n_components=5)
for i, xx in enumerate(chunked_data):
    pcs = pca.fit_transform(xx)
    plt.title(f'PCs {i+1}')
    plt.plot(pcs[:, 1])
    plt.show()


# %%
array_csi = load_array(PATH)
array_csi_modulus = abs(array_csi)  # amplitude calculating
print(array_csi_modulus.shape)
array_csi_phase = np.angle(array_csi)

drop_idx = []

for i in range(array_csi_modulus.shape[1]):
    if (np.var(array_csi_modulus[:, i]) < 1):
        drop_idx.append(i)


print(len(drop_idx))

for i in (drop_idx):
    plt.plot(array_csi_modulus[:, i])

# %%
select_list = [i for i in range(192)]


for i in drop_idx:
    select_list.remove(i)

LLTF = []
HTLTF = []
for i in select_list:
    if i < 64:
        LLTF.append(i)
    elif 64 <= i < 128:
        HTLTF.append(i)

# %%
columns = [f'sub{i}' for i in range(0, size_y)]
df_csi_modulus = pd.DataFrame(array_csi_modulus, columns=columns)
df_csi_phase = pd.DataFrame(array_csi_phase, columns=columns)

# %%
fig = px.line(df_csi_modulus, y=[
              f'sub{i}' for i in LLTF], title='t16 CSI magnitude LLTF')
fig.show()

# %%
fig = px.line(df_csi_modulus, y=[
              f'sub{i}' for i in HTLTF], title='t16 CSI magnitude HTLTF')
fig.show()

# %%
fig = px.line(df_csi_phase, y=[
              f'sub{i}' for i in LLTF], title='t16 CSI phase LLTF')
fig.show()

# %%
fig = px.line(df_csi_phase, y=[
              f'sub{i}' for i in LLTF], title='t16 CSI phase HTLTF')
fig.show()

# %%
for i in range(1, 21):
    PATH = '/home/kuntal990/projects/WiFi_Sensing_2.0/ml/exp02/room1/phone' + str(i) + '.csv'
    data = abs(load_array(PATH))
    print(data.shape)
    # plt.plot(data[:, 6])
    # plt.title('sample f{i}')
    # plt.show()

# %%
X_f = []
y_f = []

#%%
X_train2 = []
y_train2 = []

with h5py.File('/home/kuntal990/projects/WiFi_Sensing_2.0/dataset/dataset4_final.hdf5', 'r') as hf:
    X = hf.get('X')
    X_train2 = np.array(X)
    Y = hf.get('Y')
    y_train2 = np.array(Y)
hf.close()
# %%
with h5py.File('/home/kuntal990/projects/WiFi_Sensing_2.0/dataset/dataset_FINAL5.hdf5', 'w') as hf:
    X = hf.create_dataset('X', data=X_f)
    Y = hf.create_dataset('Y', data=y_f)

hf.close()
# %%
