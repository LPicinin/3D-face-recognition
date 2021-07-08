import numpy as np

from sklearn.preprocessing import MinMaxScaler
import joblib
import shutil
from Rede.models.net import Net, TerminateOnBaseline
from tensorflow.keras.callbacks import TensorBoard

# https://github.com/AISangam/Image-Augmentation-Using-OpenCV-and-Python

def getData(data: np.ndarray):
    return data['lxy'], data['lz'], data['lDescritores'], data['y_train']


train = np.load('./trainch.npz', allow_pickle=True)
test = np.load('./testch.npz', allow_pickle=True)

xy_train, z_train, descritores_train, y_train = getData(train)
xy_test, z_test, descritores_test, y_test = getData(test)

flag = True
count = 0
tensorboard = TensorBoard(log_dir="logs/fit")
while flag:
    net = Net()
    history = net.fit([xy_train, z_train, descritores_train],
                      np.array(y_train),
                      20,
                      batch_size=32,
                      callbacks=[tensorboard])

    h_loss = history.history['loss']
    h_sparse_categorical_accuracy = history.history['sparse_categorical_accuracy']

    loss, acc = net.avaliar([xy_test, z_test, descritores_test], y_test, batch_size=32)

    if acc > 0.90:
        net.getModel().save_weights(f"./pesos_avaliados/adam_{count}_{acc}_pesos.h5")
        net.getModel().save(f"./pesos_avaliados/adam_{count}_{acc}_modelo")
        count += 1
        flag = False
    else:
        shutil.rmtree('./logs/fit', ignore_errors=True)
    print('\n\n\n\n\n')

# %tensorboard --logdir logs/fit
