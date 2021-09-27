# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import argparse
import numpy as np

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

if __name__ == '__main__':
    data = load_data(PATH)
    #print(data.head())
    CSI_DATA = data['CSI_DATA']
    CSI_DATA = convert_list_string_to_numpy_int_array(CSI_DATA)
    print(CSI_DATA.shape)
