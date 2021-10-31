import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
PATH = "/Users/aayush/Aayush/WiFi_Sensing_2.0/dataset/live.csv"
def load_csi(filepath):
  df = pd.read_csv(filepath)

  df_csi = df.loc[:, ['len', 'CSI_DATA']]
  drop_idx = []
  for i in range(df_csi.shape[0]):
    if df_csi.iloc[i]['len'] < 384:
      drop_idx.append(i)

  df_csi = df_csi.drop(drop_idx)
  size_x = len(df_csi.index)
  size_y = df_csi.iloc[0]['len']//2 # no. of subcarriers ..

  array_csi = np.zeros([size_x, size_y], dtype = np.complex64)

  for x , csi in enumerate(df_csi.iloc):
      temp = csi["CSI_DATA"].replace(' ', ',')
      temp = temp.replace(',]', ']')
      csi_raw_data = json.loads(temp)
      for y in range(0, len(csi_raw_data), 2):
          array_csi[x][y//2] = complex(csi_raw_data[y], csi_raw_data[y + 1])  # IQ channel frequency response
  array_csi_modulus = abs(array_csi) 
  return array_csi_modulus

endindex = 0
while True:
    array_csi_modulus = load_csi(PATH)
    plt.plot(array_csi_modulus[endindex:])
    endindex=array_csi_modulus.shape[0]
    plt.pause(0.05)