import pandas as pd
import numpy as np
import os
import pickle
import scipy
import dateparser
import datetime
import warnings
import config_anomaly as cfg


def get_arr(values):
    print(values)

def preprocessing(df):
    df = df.set_index("Time")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[[c for c in df.columns if "S" not in c]]
    #  df = df[[c for c in df.columns if "烏日測站" not in c]]
    df.columns = [c.replace(" ", "") for c in df.columns]
    df.columns = [c.replace("L", "") for c in df.columns]
    df = df.reindex_axis(sorted(df.columns), axis=1)
    #  df['Time'] = df.index
    df = df.reset_index()
    return df


class AnomalyDetector():
    def __init__(self):
        self.coord_df = pd.read_csv(cfg.COORD_CSV_PATH)
        self.device_list = list(self.coord_df.name)
        self.coord_arr = np.array([*zip(self.coord_df.lat, self.coord_df.lon)])

        self.models = self.load_models(cfg.MODEL_PATH)

        self.record_keys = [*cfg.MEASUREMENT_LIST, 'pred', 'status', 'temp_anomaly', 'anomaly']
        self.records = self.load_records(cfg.RECORD_PATH)

        self.temp_anomaly_func = [
            ["alpha", self.detect_alpha],
            ["beta", self.detect_beta],
        ]
        self.real_anomaly_func = [
            ["small", self.detect_small],
            # ["large", self.detect_large],
        ]
        self.num_anomaly = len(self.temp_anomaly_func) + len(self.real_anomaly_func)

    def load_models(self, file_path):
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            print("No Model File")
            raise Exception

    def load_records(self, file_path):
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            records = {device: {} for device in self.device_list}
            for device in records.keys():
                for measure in self.record_keys:
                    records[device][measure] = []

        return records

    def preprocess_input(self, device, value_list):
        new_value_list = value_list.copy()
        flag = False
        for i, (measure, v) in enumerate(zip(cfg.MEASUREMENT_LIST, value_list)):
            try:
                v = float(v)
                if not np.isnan(v):
                    _min, _max = cfg.MEASUREMENT_RANGE[measure]
                    v = v if v > _min and v < _max else np.nan 
            except Exception as e:
                v = np.nan

            if np.isnan(v) and measure == cfg.OBJECTIVE:
                flag = True

            new_value_list[i] = v

        return new_value_list, flag

    def is_feat_valid(self, input_matrix):
        for feat_vec in input_matrix:
            if len(feat_vec[~np.isnan(feat_vec)]) <= 3:
                return False
        return True

    def interpolate(self, input_matrix):
        # input_arr = np.array(input_list, dtype=float)
        for feat_vec in input_matrix:
            valid_z = feat_vec[~np.isnan(feat_vec)]

            valid_x, valid_y = self.coord_arr[~np.isnan(feat_vec)].T
            nan_x, nan_y = self.coord_arr[np.isnan(feat_vec)].T
            # print(len(valid_x), len(input_arr))
            f = scipy.interpolate.interp2d(valid_x, valid_y, valid_z)

            nan_z = np.ravel([f(tx, ty) for tx, ty in zip(nan_x, nan_y)])
            feat_vec[np.isnan(feat_vec)] = nan_z
        return input_matrix

    def form_feat(self, timestamp, device):
        unshape_input = [[None] * len(cfg.MEASUREMENT_LIST) for d in self.device_list]

        for i, tmp_device in enumerate(self.device_list):
            if tmp_device == device:
                #  unshape_input[i] = [None] * len(value_list)
                continue

            rec = self.records[tmp_device]
            
            #tmp_device_df = [rec[m][-1][1] if len(rec[m]) > 0 else np.nan for m in cfg.MEASUREMENT_LIST]

            for j, measure in enumerate(cfg.MEASUREMENT_LIST):
                for k in range(1, len(rec[measure]) + 1):
                    
                # take -2 because -1 is the latest
                    t, v = rec[measure][-k]
                    #latest_time = rec[measure][0]
                    if (timestamp - t).total_seconds() > cfg.MAX_LEGAL_SECONDS:
                        break

                    if not np.isnan(v):
                        feat_key = self.record_keys[j + 1]
                        unshape_input[i][j] = v
                        break

        return np.array(unshape_input, dtype=float).T

    def detect_alpha(self, timestamp, device, true_val, pred_val):
        if np.isnan(pred_val) or np.isnan(true_val):
            return "U"
        elif true_val - pred_val > cfg.ALPHA_P_THRES:
            return "+"
        elif true_val - pred_val < cfg.ALPHA_M_THRES:
            return "-"
        return "0"

    def detect_beta(self, timestamp, device, true_val, pred_val):
        if np.isnan(true_val):
            return "U"

        for i in range(1, len(self.records[device][cfg.OBJECTIVE]) + 1):
            last_time, last_value = self.records[device][cfg.OBJECTIVE][-i]
            if (timestamp - last_time).total_seconds() > cfg.MAX_LEGAL_SECONDS:
                return "U"

            if not np.isnan(last_value):
                if true_val - last_value > cfg.BETA_THRES * (timestamp - last_time).total_seconds() / 60:
                    return "1"
                else:
                    return "0"

        return "U"

    def detect_small(self, timestamp, device, true_val, pred_val, cur_temp_anomaly):
        i = 1
        cur_alpha, cur_beta = list(cur_temp_anomaly[0:2])
        if cur_alpha == "U" or cur_beta == "U":
            return "U"

        for i in range(1, len(self.records[device]['temp_anomaly']) + 1):
            last_time, last_temp_anomaly = self.records[device]['temp_anomaly'][-i]
            if (timestamp - last_time).total_seconds() > cfg.MAX_LEGAL_SECONDS:
                return "U"
            
            last_alpha, last_beta = list(last_temp_anomaly[0:2])
            if last_alpha != "U" and last_beta != "U": 
                if (cur_alpha == '+' and cur_beta == '0' and
                        last_alpha == '+' and last_beta == '1'):
                    return "1"
                else:
                    return "0"
            i += 1
        return "U"

    def detect_large(self, timestamp, device, true_val, pred_val, temp_anomaly_code):
        return "U"

    def detect_temp_anomaly(self, timestamp, device, true_val, pred_val):
        anomaly_list = [0] * len(self.temp_anomaly_func)
        for i, (func_name, func) in enumerate(self.temp_anomaly_func):
            anomaly_list[i] = func(timestamp, device, true_val, pred_val)
        return "".join(anomaly_list)

    def detect_real_anomaly(self, timestamp, device, true_val, pred_val, temp_anomaly_code):
        anomaly_list = [0] * len(self.real_anomaly_func)
        for i, (func_name, func) in enumerate(self.real_anomaly_func):
            anomaly_list[i] = func(timestamp, device, true_val, pred_val, temp_anomaly_code)
        return "".join(anomaly_list)

    def detect(self, timestamp_str, device, value_list):
        status = "0"
        temp_anomaly = "U" * len(self.temp_anomaly_func)
        real_anomaly = "U" * len(self.real_anomaly_func)

        timestamp = dateparser.parse(str(timestamp_str))
        if timestamp is None:
            timestamp = datetime.datetime.now()

        value_list, flag = self.preprocess_input(device, value_list)
        if flag:
            status, pred = "X", np.nan
        else:
            # Build Feature
            feat_matrix = self.form_feat(timestamp, device)
            if not self.is_feat_valid(feat_matrix):
                status, pred = "I", np.nan
            elif device not in self.models:
                status, pred = "M", np.nan
            else:
                # Interpolate
                feat_matrix = self.interpolate(feat_matrix)

                # Predict
                feat_vector = np.array(feat_matrix).T.reshape(1, -1)
                pred = self.models[device].predict(feat_vector)[0]

                _min, _max = cfg.MEASUREMENT_RANGE[cfg.OBJECTIVE]
                pred = pred if pred > _min and pred < _max else np.nan

            # Detect Anomaly
            temp_anomaly = self.detect_temp_anomaly(timestamp, device, value_list[0], pred)
            real_anomaly = self.detect_real_anomaly(timestamp, device, value_list[0], pred, temp_anomaly)

        # Save Latest Record
        rec = [*value_list, pred, status, temp_anomaly, real_anomaly]
        for value, key in zip(rec, self.record_keys):
            self.records[device][key].append((timestamp, value))

        #  print(device)
        #  print(status + temp_anomaly + real_anomaly)
        #  print(self.records[device]["PM2.5"])
        return status + temp_anomaly + real_anomaly

    def load(self):
        self.models = self.load_models(cfg.MODEL_PATH)

    #  def save(self):
    #     with open(self.records_path, 'wb') as f:
    #       pickle.dump(self.records, f)


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    ad = AnomalyDetector()
    test_df = pd.read_csv("te1.csv", encoding="utf-8")
    #  test_df = preprocessing(test_df)
    test_df = test_df.set_index("Timestamp")
    for c in test_df.columns:
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

    results = {key: [] for key in test_df.columns}
    preds = {key: [] for key in test_df.columns}
    for i, rows in enumerate(test_df.iterrows()):
        if i % 1000 == 0:
            print(i)
            #  exit()
        timestamp = rows[0]
        devices = rows[1].index
        values = list(rows[1])

        tmp = []
        for device, value in zip(devices, values):
            r = ad.detect(timestamp, device, [value, value])
            results[device].append(r)
            preds[device].append(float(ad.records[device]['pred'][-1][1]))
            #  tmp.append(ad.records[device].iloc[-1])
        #  print(tmp)
    with open("aar1", "wb") as f:
        pickle.dump(results, f)
    with open("pd1", "wb") as f:
        pickle.dump(preds, f)

    #  ad.save()
    #  ad.detect("", 'A1_3M', [12, 12])
    #  ad.detect("", 'A1_3M', [20, 20])
    #  ad.detect("", '中科管理局', [20, 20])
    #  ad.detect("", '烏日測站', [30, 30])
    #  ad.detect("", '監測車', [20, 20])
    #  ad.detect("", '陽明國小', [30, 30])
    #  ad.detect("", '陽明國小', [30, 30])
#
