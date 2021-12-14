#%%
# from keras.layers import Activation, Dropout, Flatten, Dense  # %%
# from keras.layers import Activation, Dropout, Flatten, Dense
# from keras.layers import Conv2D, MaxPooling2D
# from keras.models import Sequential
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
import pywt
from sklearn.preprocessing import StandardScaler
import scaleogram as scg


#%%
data = []
labels = []

with h5py.File('/home/kuntal990/projects/WiFi_Sensing_2.0/dataset/dataset_FINAL5.hdf5', 'r') as hf:
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
scaler = StandardScaler()
new_data = np.empty(shape=(data.shape[0], data.shape[1], n_comps))
for i in range(data.shape[0]):
    tmp = scaler.fit_transform(data[i])
    pcs = pca.fit_transform(tmp)
    new_data[i] = pcs

print(new_data.shape)
#%%

#%%

def plot_chunk(x, y):
    plt.plot(x[:, 2])
    plt.title('Labelled as ' + str(y))
    plt.show()


#%%
plot_chunk(new_data[1], labels[1])


# %%
<<<<<<< HEAD
scales = np.arange(1, 65)
=======
wave1 = new_data[12, :, 1]
plot_chunk(wave1, 2)
#%%
walk1 = new_data[2, :, 1]
#plot_chunk(walk1, 1)
still1 = new_data[56, :, 1]
#plot_chunk(still1, 0)
# %%
scales = np.arange(1, 128)
>>>>>>> 1512fd97aa488237f4ceab949e9525395d1e575e
[coeff, freq] = pywt.cwt(new_data[20, :, 1], scales=scales, wavelet='mexh')


#%%
plt.imshow(coeff, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
           vmax=abs(coeff).max(), vmin=-abs(coeff).max())

# %%
scg.set_default_wavelet('morl')
scales = 64
scales = scg.periods2scales(np.arange(1, scales + 1))

#%%
k = 111
scg.cws(new_data[k, :, 1], scales=scales, figsize=(10, 4.0), coi=False, ylabel="Period", xlabel="Time",
        title='cwt plot')
print(labels[k])
# %%
scg.plot_wavelets(figsize=(15, 15))

# %%
from skimage.transform import resize

rescale_size = 128
scales = 64
X_cwt = np.ndarray(shape=(int(new_data.shape[0]), rescale_size, scales, n_comps), dtype='float32')

for i in range(len(new_data)):
    for j in range(n_comps):
        sample = new_data[i,:,j]
        coeffs, freq = pywt.cwt(sample, np.arange(1, scales+1), 'morl')
        re_coeffs = resize(coeffs, (rescale_size, scales), mode='constant')
        X_cwt[i, :, :, j] = re_coeffs




#%%
X_train, X_test, y_train, y_test = train_test_split(
    X_cwt, labels, test_size=0.40, random_state=42, stratify=labels)

print(f"shapes (n_samples, x_img, y_img, z_img) of X_train_cwt: {X_train.shape}")
#%%
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
#import keras (high level API) wiht tensorflow as backend
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
#%%
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

#%%
def build_cnn_model(activation, input_shape):
    model = Sequential()
    
    # 2 Convolution layer with Max polling
    model.add(Conv2D(32, 5, activation = activation, padding = 'same', input_shape = input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 5, activation = activation, padding = 'same', kernel_initializer = "he_normal"))
    model.add(MaxPooling2D())  
    model.add(Flatten())
    
    # 3 Full connected layer
    model.add(Dense(128, activation = activation, kernel_initializer = "he_normal"))
    model.add(Dense(54, activation = activation, kernel_initializer = "he_normal"))
    model.add(Dense(5, activation = 'softmax')) # 6 classes
    
    # summarize the model
    print(model.summary())
    return model

def compile_and_fit_model(model, X_train, y_train, X_test, y_test, batch_size, n_epochs):

    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])
    
    # define callbacks
    callbacks = [ModelCheckpoint(filepath='./checkpoints/best_model.tf', save_format='tf', monitor='val_sparse_categorical_accuracy', save_best_only=True)]
    
    # fit the model
    history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(X_test, y_test))
    
    return model, history

#%%
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

cnn_model = build_cnn_model("relu", input_shape)

#%%
trained_cnn_model, cnn_history = compile_and_fit_model(cnn_model, X_train, y_train, X_test, y_test, 1, 100)

# %%
cnn_eval = build_cnn_model("relu", input_shape)
#cnn_eval.load_weights('./checkpoints/best_model.tf')
cnn_eval.load_weights('/home/kuntal990/projects/WiFi_Sensing_2.0/Code/Expt2/checkpoints/best_model.tf/saved_model.pb')
loss, acc = cnn_eval.evaluate(X_test, y_test, verbose=2)
#%%
import seaborn as sns
from sklearn import metrics

LABEL_NAMES = ['still', 'walk', 'wave', 'page turn', 'phone picking']

def create_confusion_matrix(y_pred, y_test):    
    #calculate the confusion matrix
    confmat = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(confmat, cmap=plt.cm.Blues, alpha=0.5)

    n_labels = len(LABEL_NAMES)
    ax.set_xticks(np.arange(n_labels))
    ax.set_yticks(np.arange(n_labels))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)

    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # loop over data dimensions and create text annotations.
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=i, y=j, s=confmat[i, j], va='center', ha='center')
    
    # avoid that the first and last row cut in half
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    ax.set_title("Confusion Matrix")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    plt.tight_layout()
    plt.show()

# make predictions for test data
y_pred = trained_cnn_model.predict_classes(X_test)
# determine the total accuracy 
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

create_confusion_matrix(y_pred, y_test)
# %%
