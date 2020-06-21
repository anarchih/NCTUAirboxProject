import pickle
import pandas as pd
import numpy as np
import scipy
import warnings
import config_anomaly as cfg
import os

def preprocessing(df):
    df = df.set_index("Timestamp")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    #  df = df[[c for c in df.columns if "S" not in c]]
    #  df = df[[c for c in df.columns if "烏日測站" not in c]]
    #  df.columns = [c.replace(" ", "") for c in df.columns]
    #  df.columns = [c.replace("L", "") for c in df.columns]
    #  df = df.reindex_axis(sorted(df.columns), axis=1)
    # df['Timestamp'] = df.index
    # df = df.reset_index()
    return df


def is_input_valid(input_list):
    input_arr = np.array(input_list, dtype=float)
    if len(input_arr[~np.isnan(input_arr)]) <= 3:
        return False
    return True


def interpolate(input_list, coord_arr):
    input_arr = np.array(input_list, dtype=float)
    valid_z = input_arr[~np.isnan(input_arr)]

    valid_x, valid_y = coord_arr[~np.isnan(input_arr)].T
    nan_x, nan_y = coord_arr[np.isnan(input_arr)].T
    # print(len(valid_x), len(input_arr))
    f = scipy.interpolate.interp2d(valid_x, valid_y, valid_z)

    nan_z = np.ravel([f(tx, ty) for tx, ty in zip(nan_x, nan_y)])
    input_arr[np.isnan(input_arr)] = nan_z
    return list(input_arr)


def build_y(df, device):
    new_df = df.copy()
    #  new_df = new_df[~pd.isna(new_df[device])]
    idx = (~np.isnan(new_df[device])) & (new_df[device] > 0)
    idx = np.arange(0, len(idx))[idx]
    y = np.array(new_df[device])
    return y, idx


def build_x(df_list, device, coord_arr):
    xx_list = []
    ii_list = []
    for df in df_list:
        new_df = df.copy()
        #  new_df = new_df[~pd.isna(new_df[device])]
        new_df[device] = np.nan
        x_raw = new_df.values
        x_list = []
        idx_list = []
        for j, x_data in enumerate(x_raw):
            if is_input_valid(x_data):
                x_data = interpolate(x_data, coord_arr)
                idx_list.append(j)
            x_list.append(x_data)
        xx_list.append(x_list)
        ii_list.append(idx_list)
    x = np.transpose(np.array(xx_list), (1, 2, 0))
    x = x.reshape((x.shape[0], -1))
    return x, ii_list

def train_anomaly(data_csv_list_str):
    models = {}
    train_data_path_list = data_csv_list_str.split(cfg.SPLIT_SYMBOL)

    coord_df = pd.read_csv(cfg.COORD_CSV_PATH)
    device_list = list(coord_df.name)
    coord_arr = np.array([*zip(coord_df.lat, coord_df.lon)])

    df_list = []
    for path, measure in zip(train_data_path_list, cfg.MEASUREMENT_LIST):
    # train_data_path in train_data_path_list
        df = pd.read_csv(path, encoding="utf-8")
        df = preprocessing(df)
        df = df[device_list]


        range_ = cfg.MEASUREMENT_RANGE[measure]
        # df[df < range_[0]] = range_[0]
        # df[df > range_[1]] = range_[1]
        df[df < range_[0]] = np.nan
        df[df > range_[1]] = np.nan
        df_list.append(df)

    for device in device_list:
        print(device)
        if device not in df_list[cfg.OBJ_IDX]:
            continue
        #  if device != '西屯測站' and device != "忠明測站":
            #  continue
        y, y_idx = build_y(df_list[cfg.OBJ_IDX], device)
        x, x_idx_list = build_x(df_list, device, coord_arr)

        # remove wrong data in (x, y)
        valid_idx = list(set.intersection(*[set(i) for i in x_idx_list], set(y_idx)))
        x = x[valid_idx]
        y = y[valid_idx]

        print(len(x))
        if len(x) < cfg.MIN_TRAINING_DATA:
            continue

        # build model
        model = cfg.TRAIN_MODEL(cfg.TRAIN_MODEL_ARGS)
        model.fit(x, y)
        models[device] = model
    
    os.makedirs(cfg.FOLDER, exist_ok=True) 
    save(cfg.MODEL_PATH, models)
    #  return models


def save(models_path, models):
    try:
        with open(models_path, "rb") as f:
            old_models = pickle.load(f)
        old_models.update(models)
    except FileNotFoundError:
        old_models = models

    with open(models_path, "wb") as f:
        pickle.dump(old_models, f)


def main():
    models = train_anomaly(
        "./tr4.csv#./tr4.csv",
    )
    #  save(cfg.MODEL_PATH, models)

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    main()
