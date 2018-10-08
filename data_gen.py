import os

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def load_data():
    # (num_samples, 320, 320, 4)
    num_samples = 8144
    train_split = 0.8
    num_train = int(round(num_samples * train_split))
    num_valid = num_samples - num_train

    x_train = np.empty((num_train, 320, 320, 3), dtype=np.float32)
    y_train = np.empty((num_train, 320, 320, 3), dtype=np.float32)
    x_valid = np.empty((num_valid, 320, 320, 3), dtype=np.float32)
    y_valid = np.empty((num_valid, 320, 320, 3), dtype=np.float32)

    i_train = i_valid = 0
    for root, dirs, files in tqdm(os.walk("data", topdown=False)):
        for name in files:
            filename = os.path.join(root, name)
            bgr_img = cv.imread(filename)
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            if filename.startswith('data/train'):
                x_train[i_train, :, :, :] = rgb_img / 255.
                y_train[i_train, :, :, :] = rgb_img / 255.
                i_train += 1
            elif filename.startswith('data/valid'):
                x_valid[i_valid, :, :, :] = rgb_img / 255.
                y_valid[i_valid, :, :, :] = rgb_img / 255.
                i_valid += 1

    return x_train, y_train, x_valid, y_valid


x_train, y_train, x_valid, y_valid = load_data()


class VaeDataset(Dataset):
    def __init__(self, split):
        self.split = split

        if split == 'train':
            self.x = x_train
            self.y = y_train
        else:
            self.x = x_valid
            self.y = y_valid

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)
