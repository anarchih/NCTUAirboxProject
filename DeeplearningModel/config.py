import pandas as pd
from utils import *


class BaseConfig(object):
    day_len = 10 * 24
    start_date = "2018-03-31"
    end_date = "2019-04-30"
    total_day = len(pd.date_range(start=start_date, end=end_date, freq="1D"))
    total_len = total_day * day_len
    print(total_len)
    train_len = 24 * 7 * day_len
    test_len = 8 * 7 * day_len
    skip_len = 8 * 7 * day_len
    pad_len = 0
    train_test_split = train_test_split_tail
    spc_len = 0

    # Deep Learning Config
    res = 4


class MAConfig(object):
    day_len = 4 * 24
    start_date = "2017-02-01"
    end_date = "2017-12-01"
    total_day = len(pd.date_range(start=start_date, end=end_date, freq="1D")) - 1
    total_len = total_day * day_len
    train_len = 24 * 7 * day_len
    test_len = 4 * 7 * day_len
    skip_len = 4 * 7 * day_len
    pad_len = 0
    train_test_split = train_test_split_tail
    spc_len = 1 * 7 * day_len

    # Deep Learning Config
    res = 4


class MACompareConfig(BaseConfig):
    p = BaseConfig
    pad_len = 1 * 7 * p.day_len
