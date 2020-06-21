from sklearn.linear_model import LinearRegression


FOLDER = "tmp_anomaly_folder/"
RECORD_PATH = FOLDER + "records"
MODEL_PATH = FOLDER + "m1"
COORD_CSV_PATH = "coord.csv"


SPLIT_SYMBOL = "#"
MIN_TRAINING_DATA = 1000
TRAIN_MODEL = LinearRegression
TRAIN_MODEL_ARGS = {}


MEASUREMENT_LIST = ["PM2.5", "PM10"]
MEASUREMENT_RANGE = {
    'PM2.5': (0, 100),
    'PM10': (0, 100),
}
OBJECTIVE = "PM2.5"
OBJ_IDX = MEASUREMENT_LIST.index(OBJECTIVE)

MAX_LEGAL_SECONDS = 60 * 15
ALPHA_P_THRES = 10
ALPHA_M_THRES = -10
BETA_THRES = 10 / 6
