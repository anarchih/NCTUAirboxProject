from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import argparse
import torch.utils.data as data_utils

import resnet
import os
from utils import progress_bar, get_confidence
from torch.autograd import Variable
import numpy as np
import pickle
import config
from dataloader import *
import pandas as pd
import time 

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.c = nn.L1Loss()

    def forward(self, input, target):
        return self.c(input, target)


def adjust_lr(epoch):
    lr = 0.1
    if epoch >= 20:
        lr = 0.01
    if epoch >= 40:
        lr = 0.001
    # if epoch >= 50:
        # lr = 0.0001
    # if epoch >= 60:
        # lr = 0.00001
    return lr


def train(epoch, net, trainloader, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    total_batch_num = 0
    for batch_idx, (*inputs, targets) in enumerate(trainloader):
        batch_num = targets.size()[0]
        if batch_num == 1:
            break
        if use_cuda:
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda()
            targets = targets.cuda()

        optimizer.zero_grad()

        for i in range(len(inputs)):
            inputs[i] = Variable(inputs[i])
        targets = Variable(targets)

        # inputs1, inputs2, targets = Variable(inputs1), Variable(inputs2), Variable(targets)

        outputs = net(*inputs)
        # outputs = torch.clamp(outputs, 0, 100)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_batch_num += batch_num
        train_loss += loss.item() * batch_num
        # _, predicted = torch.max(outputs.data, 1)
        # total += targets.size(0)
        # correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f' % (train_loss/total_batch_num))

    return train_loss / total_batch_num


def test(epoch, net, testloader, test_crit):
    pred_list = []
    net.eval()
    test_loss = 0
    total_batch_num = 0
    for batch_idx, (*inputs, targets) in enumerate(testloader):
        batch_num = targets.size()[0]
        if use_cuda:
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda()
            targets = targets.cuda()

        for i in range(len(inputs)):
            inputs[i] = Variable(inputs[i], volatile=True)
        targets = Variable(targets)

        outputs = net(*inputs)
        outputs = torch.clamp(outputs, 0, 100)
        loss = test_crit(outputs, targets)

        test_loss += loss.item() * batch_num
        total_batch_num += batch_num

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f' % (test_loss/total_batch_num))
        pred_list.extend(list(outputs.cpu().data.numpy().ravel()))

    return test_loss / total_batch_num, pred_list


def train_test(arr_list, start):
    train_arr = []
    test_arr = []
    for arr in arr_list:
        tr, te = cfg.train_test_split(arr, start, cfg)
        train_arr.append(tr)
        test_arr.append(te)

    train_tor = []
    test_tor = []
    for tr, te in zip(train_arr, test_arr):
        train_tor.append(torch.from_numpy(tr))
        test_tor.append(torch.from_numpy(te))

    try:
        train_dataset = TestDataset(*train_tor, cfg) # create your datset
    except IndexError:
        return None
    trainloader = data_utils.DataLoader(train_dataset, batch_size=40, shuffle=True) # create your dataloader

    try:
        test_dataset = TestDataset(*test_tor, cfg) # create your datset
    except IndexError:
        return None
    testloader = data_utils.DataLoader(test_dataset, batch_size=40) # create your dataloader

    net = resnet.TestModel()

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = CustomLoss()
    test_crit = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

    train_loss_log = []
    test_loss_log = []

    scheduler = LambdaLR(optimizer, lr_lambda=adjust_lr)
    train_time, test_time = 0, 0
    log = {'train_loss': [], 'test_loss': []}
    for epoch in range(1, 51):
        train_start_time = time.time()
        scheduler.step()
        train_loss = train(epoch, net, trainloader, criterion, optimizer)
        train_time += time.time() - train_start_time

        test_start_time = time.time()
        test_loss, preds = test(epoch, net, testloader, test_crit)
        test_time += time.time() - test_start_time

        log['train_loss'].append(train_loss)
        log['test_loss'].append(test_loss)
    
    log['truths'] = test_dataset.y
    log['pred_vals'] = preds
    log['pred_idxs'] = test_dataset.idx
    log['train_time'] = train_time
    log['test_time'] = test_time

    return log


def load_data(device):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4467), (0.2471, 0.2435, 0.4467)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4467), (0.2471, 0.2435, 0.4467)),
    ])
    
    t = np.load("final_stmean_input/time_info.npy").astype(np.float32)

    x = np.load("final_stmean_input/" + device + "_img.npy")
    x = np.nan_to_num(x)
    x = np.clip(x, 0, 100)
    x = x.transpose((0, 3, 1, 2))
    x = x.astype(np.uint8)

    m = np.load("final_stmean_input/" + device + "_mask.npy")
    m = m.reshape((-1, 1, m.shape[1], m.shape[2]))
    m = m.astype(np.uint8)

    y = np.load("final_stmean_input/" + device + "_y.npy")
    y = np.clip(y, 0, 100)
    y = y.astype(np.float32)

    return x, t, m, y


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', '-m', type=int, help='resume from checkpoint')
parser.add_argument('--start', type=int, help='resume from checkpoint')
parser.add_argument('--device', type=str, help='resume from checkpoint')

args = parser.parse_args()
cfg = config.BaseConfig
start = args.start
end = cfg.total_len - cfg.train_len - cfg.test_len - cfg.spc_len

target_list = [
    'A1', 'A2', 'A3',
    'B1', 'B2', 'B3', 'B4',
    '中科管理局', '環資中心', '沙鹿測站', '陽明國小',
    '中科實中', '忠明測站', '都會公園', '監測車',
    '國安國小', '西屯測站', '烏日測站',
]

log = {}
for device in target_list:
    use_cuda = torch.cuda.is_available()
    arr_list = load_data(device)

    log[device] = [None] * len(range(cfg.pad_len, end, cfg.skip_len))

    for i, start in enumerate(range(cfg.pad_len, end, cfg.skip_len)):
        print(device + " " + str(start))
        log[device][i] = train_test(arr_list, start)

with open("dl_stmean.log", "wb") as f:
    pickle.dump(log, f)


