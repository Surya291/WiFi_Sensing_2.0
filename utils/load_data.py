# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='Path to the dataset to be loaded', default='dataset/test9_time.csv')

args = vars(parser.parse_args())
PATH = args['path']

def load_data(PATH):
    data = pd.read_csv(PATH) 
    return data

if __name__ == '__main__':
    data = load_data(PATH)
    print(data.head())

