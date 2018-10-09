import os
import random

import cv2 as cv
import numpy as np
import torch
from scipy.misc import imread, imresize, imsave

from config import device, save_folder, imsize
from utils import ensure_folder


def main():
    checkpoint = '{}/BEST_checkpoint.tar'.format(save_folder)  # model checkpoint
    print('checkpoint: ' + str(checkpoint))
    # Load model
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    test_path = 'data/test/'
    test_images = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.jpg')]

    num_test_samples = 10
    samples = random.sample(test_images, num_test_samples)

    imgs = torch.zeros([num_test_samples, 3, imsize, imsize], dtype=torch.float, device=device)

    ensure_folder('images')
    for i, path in enumerate(samples):
        # Read images
        img = imread(path)
        img = imresize(img, (imsize, imsize))
        imsave('images/{}_image.png'.format(i), img)

        img = img.transpose(2, 0, 1)
        assert img.shape == (3, imsize, imsize)
        assert np.max(img) <= 255
        img = torch.FloatTensor(img / 255.)
        imgs[i] = img

    imgs = torch.tensor(imgs)

    with torch.no_grad():
        preds = model(imgs)

    for i in range(num_test_samples):
        out = preds[i]
        out = out.cpu().numpy()
        out = np.transpose(out, (1, 2, 0))
        out = out * 255.
        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)
        out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
        cv.imwrite('images/{}_out.png'.format(i), out)


if __name__ == '__main__':
    main()
