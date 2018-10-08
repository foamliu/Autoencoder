###############################
## This document created by Alexandre Boulch, ONERA, France is
## distributed under GPL license
###############################
import time

import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import *

from config import *
from data_gen import VaeDataset
from models import SegNet
from utils import *


def train(epoch, train_loader, model, optimizer):
    # Ensure dropout layers are in train mode
    model.train()

    # Loss function
    # criterion = nn.MSELoss().to(device)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)
    accs = ExpoAverageMeter()  # accuracy

    start = time.time()

    # Batches
    for i_batch, (x, y) in enumerate(train_loader):
        # Zero gradients
        optimizer.zero_grad()

        # Set device options
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)

        loss = torch.sqrt((y_hat - y).pow(2).mean())
        loss.backward()

        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i_batch % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i_batch, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  loss=losses))


def valid(model):
    model.eval()

    # iteration over the batches
    batches = []
    for batch_idx, batch_files in enumerate(tqdm(batches)):
        # containers
        bs = len(batch_files)
        batch = np.zeros((bs, input_nbr, imsize, imsize), dtype=float)
        batch_labels = np.zeros((bs, imsize, imsize), dtype=int)

        # fill batches
        # ...

        data_s2 = Variable(torch.Tensor(batch))
        target = Variable(torch.LongTensor(batch_labels))

        data_s2, target = data_s2.to(device), target.to(device)

        batch_th = Variable(torch.Tensor(batch))
        target_th = Variable(torch.LongTensor(batch_labels))

        batch_th = batch_th.to(device)
        target_th = target_th.to(device)

        # predictions
        output = model(batch_th)

        # ...


def main():
    train_loader = DataLoader(dataset=VaeDataset('train'), batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=VaeDataset('valid'), batch_size=batch_size, shuffle=False,
                            pin_memory=True, drop_last=True)
    # Create SegNet model
    label_nbr = 3
    model = SegNet(label_nbr)
    model.load_weights("vgg16-00b39a1b.pth")  # load segnet weights
    model.eval()
    print(model)

    # define the optimizer
    optimizer = optim.LBFGS(model.parameters(), lr=lr)

    epochs_since_improvement = 0

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train_loss = train(epoch, train_loader, model, optimizer)
        print("train_loss " + str(train_loss))

        # validation / test
        valid(val_loader, model)


if __name__ == '__main__':
    main()
