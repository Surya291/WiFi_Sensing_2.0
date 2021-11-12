#%%
import h5py
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn import preprocessing

#%%
data = []
labels = []

with h5py.File('/home/kuntal990/projects/WiFi_Sensing_2.0/dataset/dataset1_final.hdf5', 'r') as hf:
    X = hf.get('X')
    data = np.array(X)
    Y = hf.get('Y')
    labels = np.array(Y)
hf.close()

#%%
data = data[:, :, :64]
print(data.shape)
N = data.shape[0]
#%%

def plot_chunk(x, y):
    plt.plot(x[:, 6])
    plt.title('Labelled as ' + str(y))
    plt.show()


#%%
plot_chunk(data[2], labels[2])

pca = PCA(n_components=5)
pcs = pca.fit_transform(data)
print(data.shape)

#%%
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.20, random_state=42)

# %%




#%%
