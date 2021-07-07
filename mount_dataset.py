import os
import glob

import cv2
import numpy as np
import torch
import pix2vertex as p2v
from DataAumentation import Data_augmentation

'''
pip install imageio
pip install scikit-image
pip install tqdm
pip install mediapipe
'''
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def first_chars(x):
    return x[:x.index('-') - 1]


root_dir = './Base de dados'
dest_dir = './dataset'

# Initializations
from utils import getGrossCharacteristics, getGrossCharacteristics2, getPointsFaceMesh, getPointsFaceMesh2

facemark = cv2.face.createFacemarkLBF()
facemark.loadModel('./weights/lbfmodel.yaml')
cascade = cv2.CascadeClassifier('./weights/haarcascade_frontalface_default.xml')
detector = p2v.Detector(cascade, facemark)
sift = cv2.SIFT_create()

# parece não fazer muita diferença
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

reconstructor = p2v.Reconstructor(weights_path='weights/faces_hybrid_and_rotated_2.pth', detector=detector)

images_invalidas = []
contador_invalidos = np.zeros(200)
contador_validos = np.zeros(200)
listas = []
for i in range(200):
    listas.append([])

arquivos = sorted(os.listdir(root_dir), key=first_chars)

da = Data_augmentation()

# arr = np.array(arquivos).reshape(200, 14)
# lin = arr[0]
# lin1 = arr[1]


for x in range(200):
    classe = x
    dst = f'{dest_dir}/{x + 1}'
    if not os.path.exists(dst):
        os.mkdir(dst)
    files = glob.glob(f'./Base de dados/{x + 1}-*.jpg')
    for index_file, file in enumerate(files):
        imgs = da.image_augment(cv2.imread(file, cv2.IMREAD_COLOR))

        for index, img in enumerate(imgs):
            auximg, _, _ = detector.detect_and_crop(img)
            if auximg is not None:
                img_crop = cv2.cvtColor(auximg, cv2.COLOR_BGR2RGB)
                net_res = reconstructor.run_net(img_crop)
                final = reconstructor.post_process(net_res)
                xyz, final_image = getGrossCharacteristics2(final, img_crop)

                # keypoints
                pts = getPointsFaceMesh2(final_image)
                gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
                keypoint = list(map(lambda ponto: cv2.KeyPoint(ponto[0][0], ponto[0][1], 10), pts))
                kp, des = sift.compute(gray, keypoint)

                #-------------------------------
                l_x = []
                l_y = []
                l_z_aux = []
                for p in pts:
                    pixel = xyz[p[0][0], p[0][1]]
                    l_x.append(pixel[0])
                    l_y.append(pixel[1])
                    l_z_aux.append(pixel[2])

                xy = np.dstack((np.array(l_x), np.array(l_y)))
                xy = xy.reshape((275, 2))

                dados = np.array([xy, l_z_aux, des, classe])

                # cv2.imshow('img_final', final_image)
                print(f"Gerando dado: {file} -> {dst}-{index}")
                np.savez_compressed(f"{dst}/{index_file}-{index}.npz", dados)
                # shutil.copy(f"{root_dir}/{x + 1}-{ys}.jpg", dst)
                contador_validos[x] = contador_validos[x] + 1
            else:
                contador_invalidos[x] = contador_invalidos[x] + 1
                listas[x].append(index_file)

for index in range(200):
    if contador_invalidos[index] > 0:
        print(
            f"Pessoa {index} tem {contador_invalidos[index]} fotos inválidas, imagens retiradas do dataset: -> {str(listas[index])}")

        import shutil

        shutil.move(f'{dest_dir}/{index + 1}', f'./test/{index + 1}', copy_function=shutil.copytree)

        # shutil.rmtree(f'{dest_dir}/{index + 1}')
        # shutil.copytree(f'{dest_dir}/{index + 1}',f'./test/{dest_dir}/{index + 1}', ignore_dangling_symlinks=True)
        # shutil.rmtree(f'{dest_dir}/{index + 1}')

# Passo da reconstrução
"""
import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd
import time

import pix2vertex as p2v
from sklearn.model_selection import train_test_split

# Initializations
from utils import getGrossCharacteristics

facemark = cv2.face.createFacemarkLBF()
facemark.loadModel('./weights/lbfmodel.yaml')
cascade = cv2.CascadeClassifier('./weights/haarcascade_frontalface_default.xml')
detector = p2v.Detector(cascade, facemark)

# parece não fazer muita diferença
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

reconstructor = p2v.Reconstructor(weights_path='weights/faces_hybrid_and_rotated_2.pth', detector=detector)

parent_dir_origin = "captura/database"
parent_dir_dest = os.path.join("", "dataset")
if os.path.isdir(parent_dir_dest) is False:
    os.mkdir(parent_dir_dest)

classes = os.listdir(parent_dir_origin)
classes.remove('.gitkeep')

imagens = []
x = []
y = []
for classe in classes:
    images_esquerda = sorted(glob.glob('{}/{}/*_left.png'.format(parent_dir_origin, classe), recursive=True))
    images_direita = sorted(glob.glob('{}/{}/*_right.png'.format(parent_dir_origin, classe), recursive=True))
    imagens.append({'esq': images_esquerda, 'dir': images_direita})
i = 0
for classe_index in range(len(classes)):
    path_aux = os.path.join(parent_dir_dest, classes[classe_index])
    if os.path.isdir(path_aux) is False:
        os.mkdir(path_aux)
    for index in range(len(imagens[classe_index]['esq'])):
        img_esq = cv2.imread(imagens[classe_index]['esq'][index], cv2.IMREAD_COLOR)
        img_dir = cv2.imread(imagens[classe_index]['dir'][index], cv2.IMREAD_COLOR)

        auximg, _, _ = detector.detect_and_crop(img_esq)
        img_crop_esq = cv2.cvtColor(auximg, cv2.COLOR_BGR2RGB)

        auximg, _, _ = detector.detect_and_crop(img_dir)
        img_crop_dir = cv2.cvtColor(auximg, cv2.COLOR_BGR2RGB)

        net_res_ESQ = reconstructor.run_net(img_crop_esq)
        net_res_DIR = reconstructor.run_net(img_crop_dir)

        final_ESQ = reconstructor.post_process(net_res_ESQ)
        final_DIR = reconstructor.post_process(net_res_DIR)

        _, xyzg_esq = getGrossCharacteristics(final_ESQ, img_crop_esq)
        _, xyzg_dir = getGrossCharacteristics(final_DIR, img_crop_dir)

        uni = np.concatenate((xyzg_esq, xyzg_dir))
        path = '{}/{}.npy'.format(path_aux, index)
        np.save(path, uni)
        x.append(path)
        y.append(classe_index)
        print(i)
        i = i+1

X_train = np.array([])
X_test = np.array([])
y_train = np.array([])
y_test = np.array([])

x = np.array(x)
y = np.array(y)
for i in range(len(classes)):
    mask = y == i
    X_train_aux, X_test_aux, y_train_aux, y_test_aux = train_test_split(x[mask], y[mask], test_size=0.33, random_state=42)

    X_train = np.concatenate((X_train, X_train_aux))
    X_test = np.concatenate((X_test, X_test_aux))

    y_train = np.concatenate((y_train, y_train_aux))
    y_test = np.concatenate((y_test, y_test_aux))


data_train = {
    'path': X_train,
    'classe': y_train
}
data_test = {
    'path': X_test,
    'classe': y_test
}

df_train = pd.DataFrame(data_train, columns=['path', 'classe'])
df_test = pd.DataFrame(data_test, columns=['path', 'classe'])

df_train.to_csv(os.path.join(parent_dir_dest, "train.csv"), index=False)
df_test.to_csv(os.path.join(parent_dir_dest, "test.csv"), index=False)


# for x in glob.glob('path/**/*.c', recursive=True):
# print(x)
"""
