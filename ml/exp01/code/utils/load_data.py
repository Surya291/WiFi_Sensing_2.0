# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import argparse
import numpy as np
import json
# parser = argparse.ArgumentParser()
# parser.add_argument('-p', '--path', help='Path to the dataset to be loaded', default='dataset/test9_time.csv')

# args = vars(parser.parse_args())
# PATH = args['path']

def convert_list_string_to_numpy_int_array(data):
    #data = [for x in data ]
    data = [x[1:-1] for x in data]
    data = [list(map(np.int64,x.strip().split())) for x in data]
    data = np.array(data)
    # print(data.shape)
    return data

def load_data(PATH):
    data = pd.read_csv(PATH)
    return data

def load_CSI(PATH):
    data = pd.read_csv(PATH)
    data = data['CSI_DATA']
    data = [x[1:-1] for x in data]
    data = [list(map(np.int64, x.strip().split())) for x in data]
    data = np.array(data)
    return data

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
      if(temp[-1]!= "]"):
          temp+="]"
      else:
        temp = temp.replace(',]', ']')
      csi_raw_data = json.loads(temp)
      for y in range(0, len(csi_raw_data)-1, 2):
          array_csi[x][y//2] = complex(csi_raw_data[y], csi_raw_data[y + 1])  # IQ channel frequency response
  array_csi_modulus = abs(array_csi) 
  return array_csi_modulus
if __name__ == '__main__':
    data = load_data(PATH)
    #print(data.head())
    CSI_DATA = data['CSI_DATA']
    CSI_DATA = convert_list_string_to_numpy_int_array(CSI_DATA)
    print(CSI_DATA.shape)
