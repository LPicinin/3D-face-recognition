import os
import cv2
import numpy as np
import math
from skimage.transform import resize


class Detector:
    def __init__(self, cascade, facemark):
        self.detector = cascade
        self.predictor = facemark

    def shape_to_np(pontos, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        i = 0
        for (x, y) in pontos:
            coords[i] = (x, y)
            i = i + 1
        return coords

    def detect_and_crop(self, img, img_size=512):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self.detector.detectMultiScale(gray, 1.3, 5)  # Take a single detection
        if dets is not None and len(dets) > 0:
            _, pontos_vet = self.predictor.fit(gray, dets)
            points = pontos_vet[0].astype(int)[0]  # só a primeira face

            min_x = np.min(points[:, 0])
            min_y = np.min(points[:, 1])
            max_x = np.max(points[:, 0])
            max_y = np.max(points[:, 1])
            box_width = (max_x - min_x) * 1.2
            box_height = (max_y - min_y) * 1.2
            bbox = np.array([min_y - box_height * 0.3, min_x, box_height, box_width]).astype(np.int)

            img_crop = Detector.adjust_box_and_crop(img, bbox, crop_percent=150, img_size=img_size)
            # img_crop = img[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3], :]
            return img_crop, points, dets
        else:
            return None, None, None

    @staticmethod
    def adjust_box_and_crop(img, bbox, crop_percent=100, img_size=None):
        w_ext = math.floor(bbox[2])
        h_ext = math.floor(bbox[3])
        bbox_center = np.round(np.array([bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]]))
        max_ext = np.round(crop_percent / 100 * max(w_ext, h_ext) / 2)
        top = max(1, bbox_center[0] - max_ext)
        left = max(1, bbox_center[1] - max_ext)
        bottom = min(img.shape[0], bbox_center[0] + max_ext)
        right = min(img.shape[1], bbox_center[1] + max_ext)
        height = bottom - top
        width = right - left
        # make the frame as square as possible
        if height < width:
            diff = width - height
            top_pad = int(max(0, np.floor(diff / 2) - top + 1))
            top = max(1, top - np.floor(diff / 2))
            bottom_pad = int(max(0, bottom + np.ceil(diff / 2) - img.shape[0]))
            bottom = min(img.shape[0], bottom + np.ceil(diff / 2))
            left_pad = 0
            right_pad = 0
        else:
            diff = height - width
            left_pad = int(max(0, np.floor(diff / 2) - left + 1))
            left = max(1, left - np.floor(diff / 2))
            right_pad = int(max(0, right + np.ceil(diff / 2) - img.shape[1]))
            right = min(img.shape[1], right + np.ceil(diff / 2))
            top_pad = 0
            bottom_pad = 0

        # crop the image
        img_crop = img[int(top):int(bottom), int(left):int(right), :]
        # pad the image
        img_crop = np.pad(img_crop, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), 'constant')
        if img_size is not None:
            img_crop = resize(img_crop, (img_size, img_size)) * 255
            img_crop = img_crop.astype(np.uint8)
        return img_crop