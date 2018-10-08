###############################
## This document created by Alexandre Boulch, ONERA, France is
## distributed under GPL license
###############################

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import *

from config import *
from segnet import SegNet

# cuda
USE_CUDA = torch.cuda.is_available()

# Create SegNet model
label_nbr = 45
model = SegNet(label_nbr)
model.load_weights("vgg16-00b39a1b.pth")  # load segnet weights
if USE_CUDA:  # convert to cuda if needed
    model.cuda()
else:
    model.float()
model.eval()
print(model)

# define the optimizer
optimizer = optim.LBFGS(model.parameters(), lr=lr)


def train(epoch):
    model.train()

    # update learning rate
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # define a weighted loss (0 weight for 0 label)
    weights_list = [0] + [1 for i in range(17)]
    weights = np.asarray(weights_list)
    weigthtorch = torch.Tensor(weights_list)
    if (USE_CUDA):
        loss = nn.CrossEntropyLoss(weight=weigthtorch).cuda()
    else:
        loss = nn.CrossEntropyLoss(weight=weigthtorch)

    total_loss = 0

    # iteration over the batches
    batches = []
    for batch_idx, batch_files in enumerate(tqdm(batches)):

        # containers
        batch = np.zeros((batch_size, input_nbr, imsize, imsize), dtype=float)
        batch_labels = np.zeros((batch_size, imsize, imsize), dtype=int)

        # fill the batch
        # ...

        batch_th = Variable(torch.Tensor(batch))
        target_th = Variable(torch.LongTensor(batch_labels))

        if USE_CUDA:
            batch_th = batch_th.cuda()
            target_th = target_th.cuda()

        # initilize gradients
        optimizer.zero_grad()

        # predictions
        output = model(batch_th)

        # Loss
        output = output.view(output.size(0), output.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)

        l_ = loss(output.cuda(), target)
        total_loss += l_.cpu().data.numpy()
        l_.cuda()
        l_.backward()
        optimizer.step()

    return total_loss / len(files)


def test(epoch):
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
        if USE_CUDA:
            data_s2, target = data_s2.cuda(), target.cuda()

        batch_th = Variable(torch.Tensor(batch))
        target_th = Variable(torch.LongTensor(batch_labels))

        if USE_CUDA:
            batch_th = batch_th.cuda()
            target_th = target_th.cuda()

        # predictions
        output = model(batch_th)

        # ...


for epoch in range(1, epochs + 1):
    print(epoch)

    # training
    train_loss = train(epoch)
    print("train_loss " + str(train_loss))

    # validation / test
    test(epoch)
