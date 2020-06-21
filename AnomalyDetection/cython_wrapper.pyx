from libc.stddef cimport wchar_t
from anomaly import AnomalyDetector, get_arr
from train_anomaly import train_anomaly, save
from sklearn.linear_model import LinearRegression
from cpython.ref cimport PyObject 
from cpython cimport array
import array
import config_anomaly as cfg
#  cdef wchar_t* buf = <wchar_t*>calloc(100, sizeof(wchar_t))
cdef extern from "Python.h":
    PyObject* PyUnicode_FromWideChar(wchar_t *w, Py_ssize_t size)


#  cdef public int detect(char* timestamp, wchar_t* device, float value):
    #  device_char = <char*> device
    #  print(timestamp, device_char, value)
    #  ad = AnomalyDetector("coord.csv", "tmp_anomaly_folder/")
    #  device_str = device_char.decode("utf-8")
    #  print(ad.detect(timestamp, device_str, value))
    #  return 100

ad = AnomalyDetector()

cdef public int detect(char* timestamp, wchar_t* device, float* values, char* signal):
    cdef object device_py = <object>PyUnicode_FromWideChar(device, -1)
    try:
        timestamp_py = timestamp.decode("utf-8")
        value_list = [values[i] for i in range(len(cfg.MEASUREMENT_LIST))]
        #  print(timestamp_py, device_py, value_list)
        signal_py = ad.detect(timestamp_py, device_py, value_list).encode()
        for i, s in enumerate(signal_py):
            signal[i] = s
        signal[i + 1] = '\0'
        return 0
    except:
        print("Something Wrong")
        return -1

cdef public int train(char* train_file_list_str):
    try:
        train_file_list_str_py = train_file_list_str.decode("utf-8")
        train_anomaly(train_file_list_str_py)
        return 0
    except:
        print("Something Wrong")
        return -1
    #  save(cfg, models)

cdef public int load():
    try:
        ad.load()
    except:
        print("Something Wrong")
        return -1

