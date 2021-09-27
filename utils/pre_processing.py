#%%
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
from utils.load_data import load_CSI
#%%
x = load_CSI('dataset/test9_time.csv')
print(x.shape)
x = StandardScaler().fit_transform(x)
#%%

pca = PCA(n_components=6)
components = pca.fit_transform(x[0])



# %%
