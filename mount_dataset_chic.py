import os
import glob
import numpy as np
import random as rn

classes = os.listdir('./dataset')
classes.remove('.gitkeep')
classes.remove('notas.txt')

classes = sorted(classes, key=int)

train = []
test = []


train_xy = []
train_z = []
train_descritores = []
train_y_train = []

test_xy = []
test_z = []
test_descritores = []
test_y_train = []


# random.randint(0, 3)
for classe_id, classe in enumerate(classes):
    paths_l = [glob.glob(f"./dataset/{classe}/0-*.npz"), glob.glob(f"./dataset/{classe}/1-*.npz"),
               glob.glob(f"./dataset/{classe}/2-*.npz"), glob.glob(f"./dataset/{classe}/3-*.npz")]
    test_index = rn.randint(0, 3)

    for index, paths in enumerate(paths_l):
        for path in paths:
            data = np.load(path, allow_pickle=True)['arr_0']
            data[3] = classe_id

            xy, z, descritores, id = data

            if index == test_index:
                test_xy.append(xy)
                test_z.append(z)
                test_descritores.append(descritores)
                test_y_train.append(id)
            else:
                train_xy.append(xy)
                train_z.append(z)
                train_descritores.append(descritores)
                train_y_train.append(id)
    print(f"{classe_id} selecionada")

np.savez_compressed('./testch.npz', lxy=np.array(test_xy),
                    lz=np.array(test_z),
                    lDescritores=np.array(test_descritores),
                    y_train=np.array(test_y_train))

print('Arquivos de testes salvos')

np.savez_compressed('./trainch.npz', lxy=np.array(train_xy),
                    lz=np.array(train_z),
                    lDescritores=np.array(train_descritores),
                    y_train=np.array(train_y_train))
print('Arquivos de treino salvos')
