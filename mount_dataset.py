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
        print(f"Pessoa {index} tem {contador_invalidos[index]} fotos inválidas, imagens retiradas do dataset: -> {str(listas[index])}")

        import shutil

        shutil.move(f'{dest_dir}/{index + 1}', f'./test/{index + 1}', copy_function=shutil.copytree)
