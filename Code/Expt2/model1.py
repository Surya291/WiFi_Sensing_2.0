#%%
from keras.layers import Activation, Dropout, Flatten, Dense  # %%
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import h5py
import numpy as np
from numpy.core.fromnumeric import shape
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
n_comps = 5
pca = PCA(n_components=n_comps)
new_data = np.empty(shape=(data.shape[0], data.shape[1], n_comps))
for i in range(data.shape[0]):
    pcs = pca.fit_transform(data[i])
    new_data[i] = pcs

print(data.shape)

#%%

def plot_chunk(x, y):
    plt.plot(x[:, 2])
    plt.title('Labelled as ' + str(y))
    plt.show()


#%%
plot_chunk(new_data[2], labels[2])


#%%
X_train, X_test, y_train, y_test = train_test_split(
    new_data, labels, test_size=0.20, random_state=42)

# %%



#%%