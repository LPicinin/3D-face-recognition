# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import numpy
import numpy as np  # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
from pix2vertex import Detector

'''
Image Augmentation is the common used technique to improve the performance of computer vision system. 
Refer to the W2 of Convolutional Neutral Network Course on Cousera. 
specially in the WIDS dataset, which is an unbalanced dataset. 
Upsampling the images with oil-palm is the way to handle the unbalanced problem. 
Image augmentation artificially creates training images through different ways of processing or combination of multiple processing, 
such as mirroring, random rotation, shifts, shear and flips, etc. 
Keras has keras.preprocessing.image.ImageDataGenerator function to do image augmentation. Here showed how to use OpenCV to rotate, flip, and add Gaussian noise to original images.

Reference: 
https://towardsdatascience.com/image-augmentation-examples-in-python-d552c26f2873
https://medium.com/@thimblot/data-augmentation-boost-your-image-dataset-with-few-lines-of-python-155c2dc1baec

'''

import cv2
import random


class Data_augmentation:
    def __init__(self):
        self.value_brilho = 5

    def rotate(self, image, angle=0, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        w = image.shape[1]
        h = image.shape[0]
        # rotate matrix
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        # rotate
        image = cv2.warpAffine(image, M, (w, h))
        return image

    def flip(self, image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image

    def brightness(self, img, value=0):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if value > 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        elif value < 0:
            lim = abs(value) - 1
            v[v <= lim] = 0
            v[v > lim] = v[v >= lim] + value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def image_augment(self, image: numpy.ndarray, detector: Detector = None):
        images = []
        img = image.copy()
        for current_angle in range(-4, 5):
            aux = self.rotate(img, angle=current_angle)
            images.append(aux)
        size = len(images)
        for i in range(size):
            images.append(self.brightness(images[i], self.value_brilho))
            images.append(self.brightness(images[i], -self.value_brilho))

        # cv2.imwrite(save_path + '%s' % str(name_int) + '_vflip.jpg', img_flip)
        # cv2.imwrite(save_path + '%s' % str(name_int) + '_rot.jpg', img_rot)
        # cv2.imwrite(save_path + '%s' % str(name_int) + '_GaussianNoise.jpg', img_gaussian)
        if detector is not None:
            for img_valid in images:
                _, _, rec = detector.detect_and_crop(img_valid)
                if rec is None:
                    return images, False
            return images, True
        return images

    def main(file_dir, output_path):
        for root, _, files in os.walk(file_dir):
            print(root)
        for file in files:
            raw_image = Data_augmentation(root, file)
            raw_image.image_augment(output_path)
