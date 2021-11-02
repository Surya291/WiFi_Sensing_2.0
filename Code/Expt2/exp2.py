#%%
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
from numba import jit
import plotly.express as px

# %%
PATH = '/home/kuntal990/projects/WiFi_Sensing_2.0/dataset/room2/v2/t16.csv'
# %%
def plot_rssi(PATH=PATH):
  df = pd.read_csv(
      PATH)
  df_rssi = df.loc[:, ['rssi']]
  df_time = df.loc[:, ['real_timestamp']]
  #print(df_rssi)
  #print(df_time)
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
      #print(temp)
      ll = 0
      if temp[-1] != ']':
        temp = temp + ']'
        ll = -1
      temp = temp.replace(',]', ']')
      temp = temp.replace('-]', ']')
      #print((temp))
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
      tmp[i] = tmp[i] + (cumsum[i] - cumsum[max(i-N, 0)])/ float(N)
      
      tmp[i] = tmp[i] + (cumsum[min(i+N, len(x)-1)] - cumsum[i]) / float(N)
    return tmp

def remove_offset(x, w = 200):
  cumsum = np.cumsum(x, axis=1)
  for i in range(0, len(x)-1, w):
    #print(i)
    start = i
    end = min(i + w, len(x)-1)
    offset = (cumsum[end] - cumsum[start])/float(end - start)
    x[start:end] = x[start:end] - offset
  return x

#%%
PATH = '/mnt/d/wifiDataset/exp1/walk2.csv'
#plot_rssi(PATH)
data = abs(load_array(PATH))
plt.plot(data[:, 6][:500])
plt.title('Without SMA')
plt.show()
data = remove_offset(data, 200)
plt.plot(data[:,6][:500])
plt.title('With SMA')
#plt.ylim(-0.5, 0.5)
plt.show()

#%%
data = StandardScaler().fit_transform(data)
chunked_data = np.array_split(data, 12)
for i, xx in enumerate(chunked_data):
  plt.title(f'chunk {i+1}')
  plt.plot(xx[:,6])
  plt.show()

#%%
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

#%%
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
