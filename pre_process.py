# -*- coding: utf-8 -*-

import os
import random
import tarfile

import cv2 as cv
import numpy as np
import scipy.io
from tqdm import tqdm

from config import imsize


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_train_data(fnames, bboxes):
    src_folder = 'data/cars_train'
    num_samples = len(fnames)

    train_split = 0.8
    num_train = int(round(num_samples * train_split))
    train_indexes = random.sample(range(num_samples), num_train)
    print('train_indexes: '.format(str(train_indexes)))

    for i in tqdm(range(num_samples)):
        fname = fnames[i]
        (x1, y1, x2, y2) = bboxes[i]
        src_path = os.path.join(src_folder, fname)
        src_image = cv.imread(src_path)
        height, width = src_image.shape[:2]
        # margins of 16 pixels
        margin = 16
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        # print(fname)

        if i in train_indexes:
            dst_folder = 'data/train'
        else:
            dst_folder = 'data/valid'
        dst_path = os.path.join(dst_folder, fname)
        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(dst_path, dst_img)
    print('\n')


def save_test_data(fnames, bboxes):
    src_folder = 'data/cars_test'
    dst_folder = 'data/test'
    num_samples = len(fnames)

    for i in tqdm(range(num_samples)):
        fname = fnames[i]
        (x1, y1, x2, y2) = bboxes[i]
        src_path = os.path.join(src_folder, fname)
        src_image = cv.imread(src_path)
        height, width = src_image.shape[:2]
        # margins of 16 pixels
        margin = 16
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        # print(fname)

        dst_path = os.path.join(dst_folder, fname)
        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(dst_path, dst_img)
    print('\n')


def process_data(usage):
    print("Processing {} data...".format(usage))
    cars_annos = scipy.io.loadmat('data/devkit/cars_{}_annos'.format(usage))
    annotations = cars_annos['annotations']
    annotations = np.transpose(annotations)

    fnames = []
    bboxes = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        if usage == 'train':
            class_id = annotation[0][4][0][0]
            fname = annotation[0][5][0]
        else:
            fname = annotation[0][4][0]
        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        fnames.append(fname)

    if usage == 'train':
        save_train_data(fnames, bboxes)
    else:
        save_test_data(fnames, bboxes)


if __name__ == '__main__':
    # parameters
    img_width, img_height = imsize, imsize

    print('Extracting data/cars_train.tgz...')
    # if not os.path.exists('data/cars_train'):
    with tarfile.open('data/cars_train.tgz', "r:gz") as tar:
        tar.extractall('data')
    print('Extracting data/cars_test.tgz...')
    # if not os.path.exists('data/cars_test'):
    with tarfile.open('data/cars_test.tgz', "r:gz") as tar:
        tar.extractall('data')
    print('Extracting data/car_devkit.tgz...')
    # if not os.path.exists('data/devkit'):
    with tarfile.open('data/car_devkit.tgz', "r:gz") as tar:
        tar.extractall('data')

    cars_meta = scipy.io.loadmat('data/devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)
    print('class_names.shape: ' + str(class_names.shape))
    print('Sample class_name: [{}]'.format(class_names[8][0][0]))

    ensure_folder('data/train')
    ensure_folder('data/valid')
    ensure_folder('data/test')

    process_data('train')
    process_data('test')
