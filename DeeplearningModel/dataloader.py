import torch
import torch.utils.data as data_utils
import numpy as np
from utils import running_mean


class TestDataset(data_utils.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x1, x2, mask, y, cfg):
        self.cfg = cfg
        self.idx = [i for i, v in enumerate(y[self.cfg.spc_len:]) if 100 > v > 0]

        self.x1 = x1[self.cfg.spc_len:][self.idx]
        self.x2 = x2[self.cfg.spc_len:][self.idx]

        self.mask = mask[self.cfg.spc_len:][self.idx]

        self.img_size = self.x1[0].size()
        self.y = y[self.cfg.spc_len:][self.idx]


    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img_size = self.img_size
        base = i + self.cfg.spc_len

        cur = self.x1[base]
        m = self.mask[base]
        n = (cur == 0)
        # prd = self.x1[base-day_len:base:day_len]
        # trd = self.x1[base-week_len:base:week_len]

        cur = cur.view((-1, img_size[1], img_size[2]))
        m = m.view((-1, img_size[1], img_size[2]))
        n = m.view((-1, img_size[1], img_size[2]))
        # prd = prd.view((-1, img_size[1], img_size[2]))
        # trd = trd.view((-1, img_size[1], img_size[2]))
        # img = torch.cat([self.x1[i], self.x1[i + 1]], 0)

        cur = cur.type(torch.cuda.FloatTensor)
        m = m.type(torch.cuda.FloatTensor)
        n = n.type(torch.cuda.FloatTensor)
        cur = torch.cat([cur, m, n], 0)
        # other = torch.cat([self.x2[base], self.x3[base], self.x4[base]], 0)
        other = self.x2[base]
        return cur, other, self.y[base]


class TestPredDataset(data_utils.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x1, x2, mask, y, cfg):
        self.cfg = cfg

        self.x1 = x1
        self.x2 = x2

        self.mask = mask

        self.img_size = self.x1[0].size()
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img_size = self.img_size
        base = i + self.cfg.spc_len

        cur = self.x1[base]
        m = self.mask[base]
        n = (cur == 0)
        # prd = self.x1[base-day_len:base:day_len]
        # trd = self.x1[base-week_len:base:week_len]

        cur = cur.view((-1, img_size[1], img_size[2]))
        m = m.view((-1, img_size[1], img_size[2]))
        n = m.view((-1, img_size[1], img_size[2]))
        # prd = prd.view((-1, img_size[1], img_size[2]))
        # trd = trd.view((-1, img_size[1], img_size[2]))
        # img = torch.cat([self.x1[i], self.x1[i + 1]], 0)

        cur = cur.type(torch.cuda.FloatTensor)
        m = m.type(torch.cuda.FloatTensor)
        n = n.type(torch.cuda.FloatTensor)
        cur = torch.cat([cur, m, n], 0)
        # other = torch.cat([self.x2[base], self.x3[base], self.x4[base]], 0)
        other = self.x2[base]
        return cur, other, self.y[base]


class MADataset(data_utils.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x1, x2, y, cfg):
        self.cfg = cfg
        self.idx = [i for i, v in enumerate(y[self.cfg.spc_len:]) if v > 0]
        self.x1 = x1[self.cfg.spc_len:][self.idx]
        self.d_ma = running_mean(x1, cfg.day_len)[self.idx]
        self.w_ma = running_mean(x1, cfg.day_len * 7)[self.idx]
        self.x2 = x2[self.cfg.spc_len:][self.idx]
        self.img_size = self.x1[0].size()
        self.y = y[self.cfg.spc_len:][self.idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img_size = self.img_size

        cur = self.x1[i].type(torch.cuda.FloatTensor)
        d_ma = self.d_ma[i].type(torch.cuda.FloatTensor)
        w_ma = self.w_ma[i].type(torch.cuda.FloatTensor)
        # print(self.x1[base - self.cfg.day_len: base])
        # d_ma = torch.mean(self.x1[base - self.cfg.day_len:base].type(torch.cuda.FloatTensor), 0)
        # w_ma = torch.mean(self.x1[base - self.cfg.day_len * 7:base].type(torch.cuda.FloatTensor), 0)

        cur = cur.view((-1, img_size[1], img_size[2]))
        d_ma = d_ma.view((-1, img_size[1], img_size[2]))
        w_ma = d_ma.view((-1, img_size[1], img_size[2]))
        # w_ma = w_ma.view((-1, img_size[1], img_size[2]))
        img = torch.cat([cur, d_ma, w_ma], 0)
        # img = img.type(torch.cuda.FloatTensor)
        return img, self.x2[i], self.y[i]
