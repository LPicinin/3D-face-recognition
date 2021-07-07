import numpy as np

from sklearn.preprocessing import MinMaxScaler
import joblib
import shutil
from Rede.models.net import Net, TerminateOnBaseline
from tensorflow.keras.callbacks import TensorBoard

# https://github.com/AISangam/Image-Augmentation-Using-OpenCV-and-Python
# 4, 5, 6, 11, 12

"""
dados = []
classes = os.listdir('./dataset')
classes.remove('.gitkeep')
classes.remove('notas.txt')
# classes.remove('pessoas_desconhecidas')

classes = sorted(classes, key=int)

for classe_id, classe in enumerate(classes):
    paths = glob.glob(f"./dataset/{classe}/*.npz")
    l_aux = []
    for path in paths:
        data = np.load(path, allow_pickle=True)['arr_0']
        # xy, z, descritores, _ = data
        data[3] = classe_id

        # x_data.append([xy, z, descritores])
        dados.append(data)
    print(classe)
print('Salvando dados')

np.savez_compressed('./dados.npz', data=np.array(dados))

print('Dados salvos')
"""
# ----------------------------------------------------

"""
dados = np.load('./dados.npz', allow_pickle=True)['data']
print('Dados carregados')

xy = []
z = []
descritores = []
y_train = []

for dado in dados:
    auxxy, auxz, auxdescritores, auxYtrain = dado
    xy.append(auxxy)
    z.append(auxz)
    descritores.append(auxdescritores)
    y_train.append(auxYtrain)

dados = None

xy = np.array(xy)
z = np.array(z)
descritores = np.array(descritores)

y_train = np.array(y_train).astype(np.int)

aux_indices = np.arange(start=0, stop=len(z), step=1)

from sklearn.model_selection import train_test_split

arr = np.array_split(aux_indices, 198)
train_indexs = []
test_indexs = []

print('\n\n')
for i in range(198):
    train, test = train_test_split(arr[i], test_size=0.5, shuffle=True)# , random_state=42
    train_indexs = train_indexs + train.tolist()
    test_indexs = test_indexs + test.tolist()
    print(f'separando conjuntos teste - treino da classe {i}')

print('\n\n')

train_xy = xy[train_indexs]
train_z = z[train_indexs]
train_descritores = descritores[train_indexs]
train_y_train = y_train[train_indexs]

test_xy = xy[test_indexs]
test_z = z[test_indexs]
test_descritores = descritores[test_indexs]
test_y_train = y_train[test_indexs]
print('Arrays separados')
# unique, counts = np.unique(y_train, return_counts=True)
# print(counts)
#
# unique, counts = np.unique(train_y_train, return_counts=True)
# print(counts)
#
# unique, counts = np.unique(test_y_train, return_counts=True)
# print(counts)


np.savez_compressed('./test50.npz', lxy=np.array(test_xy),
                    lz=np.array(test_z),
                    lDescritores=np.array(test_descritores),
                    y_train=np.array(test_y_train))

print('Arquivos de testes salvos')

np.savez_compressed('./train50.npz', lxy=np.array(train_xy),
                    lz=np.array(train_z),
                    lDescritores=np.array(train_descritores),
                    y_train=np.array(train_y_train))
print('Arquivos de treino salvos')

# ----------------------------------------------------
"""
"""

# escurecer 5 e 10%
# clarear 5 e 10%
# inclinar
"""

'''
'''


def getData(data: np.ndarray):
    return data['lxy'], data['lz'], data['lDescritores'], data['y_train']


train = np.load('./trainch.npz', allow_pickle=True)
test = np.load('./testch.npz', allow_pickle=True)

xy_train, z_train, descritores_train, y_train = getData(train)
xy_test, z_test, descritores_test, y_test = getData(test)

flag = True
count = 0
tensorboard = TensorBoard(log_dir="logs/fit2")
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
        shutil.rmtree('./logs/fit2', ignore_errors=True)
    print('\n\n\n\n\n')

# %tensorboard --logdir logs/fit
