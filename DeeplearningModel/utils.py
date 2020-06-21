

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from geopy.distance import vincenty


def coord_to_xy(coord_df):
    lb_coord = min(coord_df['lat']), min(coord_df['lon'])
    x, y = [], []
    for lat, lon in zip(coord_df['lat'], coord_df['lon']):
        lat_d = vincenty(lb_coord, (lat, lb_coord[1])).km
        lon_d = vincenty(lb_coord, (lb_coord[0], lon)).km
        x.append(lon_d)
        y.append(lat_d)
    return x, y


def xy_to_grid(coord_df, res_x, res_y):
    grid_x, grid_y = [], []
    max_x, max_y = max(coord_df['x']), max(coord_df['y'])
    dx = max_x / (res_x - 1)
    dy = max_y / (res_y - 1)
    for x, y in zip(coord_df['x'], coord_df['y']):
        grid_x.append(int((x + dx / 2) / dx))
        grid_y.append(int((y + dy / 2) / dy))
    return grid_x, grid_y


def get_device_map(coord_df, target, width, height, rad):
    d_map = np.zeros((height, width))

    coord_df['x'], coord_df['y'] = coord_to_xy(coord_df)
    coord_df['g_x'], coord_df['g_y'] = xy_to_grid(coord_df, width, height)

    t_gx = coord_df[coord_df.device_id == target]['g_x']
    t_gy = coord_df[coord_df.device_id == target]['g_y']

    gx = coord_df[coord_df.device_id != target]['g_x']
    gy = coord_df[coord_df.device_id != target]['g_y']
    gx = [x for x in gx if t_gx - rad <= x <= t_gx + rad]
    gy = [y for y in gy if t_gy - rad <= y <= t_gy + rad]

    d_map[gy, gx] = 1
    return d_map


def get_confidence(coord_df, target, x, y, width, height, rad):
    coord_df['x'], coord_df['y'] = coord_to_xy(coord_df)
    coord_df['g_x'], coord_df['g_y'] = xy_to_grid(coord_df, width, height)

    t_gx = coord_df[coord_df.device_id == target]['g_x'].tolist()[0]
    t_gy = coord_df[coord_df.device_id == target]['g_y'].tolist()[0]

    gx = coord_df[coord_df.device_id != target]['g_x'].tolist()
    gy = coord_df[coord_df.device_id != target]['g_y'].tolist()

    new_gx, new_gy = [], []
    for xx, yy in zip(gx, gy):
        if t_gx - rad <= xx <= t_gx + rad & t_gy - rad <= yy <= t_gy + rad:
            new_gx.append(xx)
            new_gy.append(yy)
    print("Num of Neighbors: %d" % len(new_gx))
    if len(new_gx) == 0:
        return np.ones(y.shape)
    else:
        y_dist = np.mean(x[:, 0, new_gx, new_gy], axis=1).reshape((-1, 1))
        return (1 - np.abs(y - y_dist) / 100)



def train_test_split_middle(x, start, cfg):
    t1 = start + int(cfg.train_len / 2)
    t2 = t1 + cfg.test_len
    t3 = start + cfg.train_len + cfg.test_len
    train_x = x[list(range(cfg.pad_len, t1)) + list(range(t2, t3))]
    test_x = x[range(t1, t2)]
    return train_x, test_x


def train_test_split_tail(x, start, cfg):
    train_end = start + cfg.train_len + cfg.spc_len
    test_start = start + cfg.train_len
    test_end = train_end + cfg.test_len
    train_x, test_x = x[start:train_end], x[test_start:test_end]
    return train_x, test_x


def running_mean(x, N):
    x = x.data.numpy().astype(np.float)
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    x = (cumsum[N:] - cumsum[:-N]) / N
    return torch.from_numpy(x)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
