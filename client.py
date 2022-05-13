import os
import flwr as fl
import tensorflow as tf

# imports for downloading dataset
import urllib2
import tarfile
import shutil
import glob
from examples.util import get_tmd_dataset

# imports for TMDataset
import csv
import logging
import re
from os import listdir
import pandas as pd
import datasets.const as const
import datasets.util as util
import sys
import math
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import RobustScaler


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


'''Load model'''
#model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model = tmd
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

'''Load dataset'''
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Constant files for dataset download
dataset_dir = './TransportationData'
datasetBalanced = dataset_dir + '/datasetBalanced'
rawOriginaldata = dataset_dir + '/_RawDataOriginal'
url_list = ['http://cs.unibo.it/projects/us-tm2017/static/dataset/extension/5second/dataset_5secondWindow.csv',
            'http://cs.unibo.it/projects/us-tm2017/static/dataset/extension/halfsecond/dataset_halfSecondWindow.csv',
            'http://cs.unibo.it/projects/us-tm2017/static/dataset/raw_data/raw_data.tar.gz']
dataset5second = 'dataset_5secondWindow.csv'
datasethalfsecond = 'dataset_halfSecondWindow.csv'
rawdataorig = "raw_data.tar.gz"

# Define TMDclient dataset
class TMDataset:
    """
    Class that implements all preprocessing steps used in TMD Dataset
    """

    EXTRACTED_FEATURE_KEYS = [
        "#mean", "#median", "#min", "#max", "#std", "#skewness", "#kurtosis", "#ptp", "#q1", "#q3", "#iqrange",
        "#var", "#entropy", "#fft_highest_magnitude", "#fft_highest_magnitude_frequency", "#fft_total_spectrum",
        "#fft_spectral_density", "#fft_spectral_entropy", "#fft_spectral_centroid", "#fft_total_phase"
    ]

    FEATURE_KEY = 'feature_key'
    PREVIOUS_LIST_KEY = 'previous_list'
    CURRENT_LIST_KEY = 'current_list'
    TRAVEL_MODE_COLUMN = 'target'
    USER_ID_COLUMN = 'user'
    BEST_FEATURES_OUTPUT_FILE = 'best_features.csv'
    BEST_SIGNALS_OUTPUT_FILE = 'best_signals.csv'

    def __init__(self):
        """
        Initialize preprocessing variables and parameters
        """
        self.tm = []
        self.users = []
        self.sensors = []
        self.n_files = 0
        self.header = {}
        self.header_with_features = {}
        self.balance_time = 0  # in seconds
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.cv = pd.DataFrame()

    # ------------------------------------------------------------ METHODS ------------------------------------------- #

    def create_balanced_dataset(self, split_dataset=True):
        """
        Create balanced dataset using downsampling
        :param split_dataset:
        :return:
        """

        # create dataset from files
        self.create_dataset()

        # set save dirs
        dir_src = const.DIR_DATASET
        file_src = const.FILE_DATASET
        file_dst = const.FILE_DATASET_BALANCED

        # save dataset before balancing
        if not os.path.exists(dir_src):
            self.create_dataset()

        # load data from dataset in memory
        if len(self.users) == 0 or len(self.sensors) == 0 or len(self.tm) == 0:
            self.fill_data_structure()

        print("START CREATE BALANCED DATASET....")
        df = pd.read_csv(dir_src + "/" + file_src)

        # get minimum number of samples per transportation mode
        min_windows = df.shape[0]
        for t in self.tm:
            df_t = df.loc[df['target'] == t]
            if df_t.shape[0] <= min_windows:
                min_windows = df_t.shape[0]

        # group samples by user and mode
        target_df = df.groupby(['target', 'user']).agg({'target': 'count'})
        target_df['percent'] = target_df.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
        target_df.loc[:, 'num'] = target_df.apply(lambda row: util.to_num(row, min_windows), axis=1)

        # build new dataset with same number of samples for each mode and user through downsampling
        self.balance_time = min_windows
        dataset_incremental = pd.DataFrame(columns=df.columns)
        for index, rows in target_df.iterrows():
            current_df = df.loc[(df['user'] == index[1]) & (df['target'] == index[0])]
            if current_df.shape[0] == rows['num']:
                dataset_incremental = pd.concat([dataset_incremental, current_df])
            else:
                df_curr = current_df.sample(n=int(rows['num']))  # down sampling
                dataset_incremental = pd.concat([dataset_incremental, df_curr])

        # save balanced dataset to csv
        dataset_incremental.to_csv(dir_src + '/' + file_dst, index=False)
        if split_dataset:
            self.split_dataset(dataset_incremental)

        print("END CREATE BALANCED DATASET....")

    def create_dataset(self):
        """
        Create dataset
        :return:
        """
        dir_src = const.DIR_RAW_DATA_FEATURES
        dir_dst = const.DIR_DATASET
        file_dst = const.FILE_DATASET

        # create files with time window if not exsist
        if not os.path.exists(dir_src):
            self.create_time_files()

        if not os.path.exists(dir_dst):
            os.makedirs(dir_dst)
            filenames = listdir(dir_src)
            result_file_path = os.path.join(dir_dst, file_dst)
            with open(result_file_path, 'w') as result_file:
                j = 0
                for file in filenames:
                    if file.endswith(".csv"):
                        current_file_path = os.path.join(dir_src, file)
                        with open(current_file_path) as current_file:  # tm file
                            i = 0
                            for line in current_file:
                                # if the current line is not the first, the header
                                if i != 0:
                                    result_file.write(line)
                                else:
                                    if j == 0:
                                        result_file.write(line)
                                    i += 1
                            j += 1
        # else:
        #     shutil.rmtree(dir_dst)
        #     os.makedirs(dir_dst)

    def get_best_raw_signals(self, include_indicators=True, **kwargs):
        """
        Retrieves list of raw signal features headers
        :return:
        """

        # check if best features have been selected before
        best_signal_path = const.DIR_RAW_DATASET + "/" + self.BEST_SIGNALS_OUTPUT_FILE
        if os.path.isfile(best_signal_path) is False:

            # load raw signal headers
            raw_signal_features = self.get_raw_signals()

            # load preprocessed dataset
            dataframe = self.get_preprocessed_dataset()
            features_dataframe = dataframe[raw_signal_features]
            classes_dataframe = dataframe[self.TRAVEL_MODE_COLUMN]

            # execute feature selection with RFE
            feature_columns = self._get_best_sensor_features_with_rfe(
                features_dataframe=features_dataframe,
                classes_dataframe=classes_dataframe,
                **kwargs
            )
            features_dataframe = dataframe[feature_columns]
            n_features = len(feature_columns)

            # execute dimensionality reduction with KBest
            max_features = int(math.sqrt(len(dataframe)))  # rule of thumb
            feature_columns = self._get_best_sensor_features_with_kbest(
                features_dataframe=features_dataframe,
                classes_dataframe=classes_dataframe,
                k_features=min(n_features, max_features),
                **kwargs
            )

            # include indicators
            raw_signal_features = list()
            for column in feature_columns:
                raw_signal_features.append(column)
                if include_indicators:
                    raw_signal_features.append(
                        column.replace('#0', '#1')
                    )  # 0 = signal, 1 = missing indicator

            # save selected features names in file
            with open(best_signal_path, mode='x', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(['feature_header_column'])
                for column in raw_signal_features:
                    csvwriter.writerow([column])

        # retrieve best features lists
        return pd.read_csv(best_signal_path).values.flatten().tolist()

    def get_raw_signals(self):
        """
        Return raw signal columns
        :return:
        """

        # load headers
        if len(self.header) == 0:
            self.fill_data_structure()

        raw_signal_features = list()
        for i in self.header:
            if 'activityrecognition' not in self.header[i] and self.header[i] not in ['target', 'user', 'time']:
                raw_signal_features.append(self.header[i])
        return raw_signal_features

    def create_raw_dataset(self):
        """
        Create dataset of raw signals and indicators of abscence of presence of signal
        :return:
        """

        dir_src = const.DIR_RAW_DATA_HEADER
        dir_dst = const.DIR_RAW_DATASET
        file_dst_forward = const.FILE_RAW_DATASET_FORWARD
        file_dst_zero = const.FILE_RAW_DATASET_ZERO
        self.create_header_files_if_not_exist(dir_src)
        self.create_or_reset_dir(dir_dst)
        file_path_forward = os.path.join(dir_dst, file_dst_forward)
        file_path_zero = os.path.join(dir_dst, file_dst_zero)
        with open(file_path_forward, 'w') as file_forward, open(file_path_zero, 'w') as file_zero:

            # write string header
            header_string = ""
            for i in self.header:
                if 'activityrecognition' not in self.header[i]:
                    header_string += self.header[i] + ","
                    if self.header[i] not in ['target', 'user', 'time']:
                        header_string += self.header[i].replace('#0', '#1') + ","  # #0 = signal, #1 = missing indicator
            header_string = header_string[:-1]
            header_string += ",target,user\n"
            file_zero.write(header_string)
            file_forward.write(header_string)

            # write raw signals and indicators
            filenames = listdir(dir_src)
            for file in filenames:
                if file.endswith(".csv"):
                    current_file_path = os.path.join(dir_src, file)
                    df_file = pd.read_csv(current_file_path, dtype=const.DATASET_DATA_TYPE)
                    df_file_aux = df_file.copy()
                    df_file_aux = df_file_aux.fillna(0)
                    previous_row = dict()
                    for index, row in df_file.iterrows():
                        zero_row_string = ""
                        forward_row_string = ""
                        for column in df_file.columns:
                            if column in ['target', 'user', 'time']:
                                zero_row_string += str(row[column]) + ","
                                forward_row_string += str(row[column]) + ","
                            elif column not in ['activityrecognition#0', 'activityrecognition#1']:
                                signal = row[column]
                                if pd.isna(signal):
                                    missing = 1
                                    zero_signal = 0
                                    if previous_row.get(column, False):
                                        forward_signal = previous_row[column]
                                    else:
                                        forward_signal = df_file_aux[column].median()
                                else:
                                    missing = 0
                                    zero_signal = signal
                                    forward_signal = signal
                                zero_row_string += str(zero_signal) + "," + str(missing) + ","
                                forward_row_string += str(forward_signal) + "," + str(missing) + ","
                                previous_row[column] = forward_signal
                        file_zero.write(zero_row_string[:-1] + "\n")
                        file_forward.write(forward_row_string[:-1] + "\n")

    def create_header_files_if_not_exist(self, dir_src):
        # create files with header if not exist
        if not os.path.exists(dir_src):
            self.create_header_files()
        if len(self.header) == 0 or len(self.header_with_features) == 0:
            self.fill_data_structure()

    def clean_files(self):
        """
        Fix original raw files problems:
        (1)delete measure from  **sensor_to_exclude**
        (2)if **sound** or **speed** measure rows have negative time --> use module
        (3)if **time** have incorrect values ("/", ">", "<", "-", "_"...) --> delete file
        (4)if file is empty --> delete file
        :return:
        """
        if os.path.exists(const.CLEAN_LOG):
            os.remove(const.CLEAN_LOG)

        pattern_negative = re.compile("-[0-9]+")
        pattern_number = re.compile("[0-9]+")

        # create directory for correct files
        self.create_or_reset_dir(const.DIR_RAW_DATA_CORRECT)

        # create log file
        logging.basicConfig(filename=const.CLEAN_LOG, level=logging.INFO)
        logging.info("CLEANING FILES...")
        print("CLEAN FILES...")
        filenames = listdir(const.DIR_RAW_DATA_ORIGINAL)
        # iterate on files in raw data directory - delete files with incorrect rows
        n_files = 0
        deleted_files = 0
        for file in filenames:

            if file.endswith(".csv"):
                n_files += 1
                # to_delete be 1 if the file have to be excluded from the dataset
                to_delete = 0

                with open(os.path.join(const.DIR_RAW_DATA_ORIGINAL, file), 'rb') as current_file:
                    res_file_path = os.path.join(const.DIR_RAW_DATA_CORRECT, file)
                    with open(res_file_path, "w") as file_result:
                        lines = current_file.readlines()  # workaround for non-utf-8 characters
                        for line in lines:
                            try:
                                line = line.decode('utf-8')
                            except UnicodeDecodeError:
                                print("line", line, "ignored!")
                                continue
                            line_data = line.split(",")

                            if line_data[1] == "activityrecognition":
                                line_data[0] = "0"

                            end_line = ",".join(line_data[2:])
                            # check if time data is correct, if is negative, make modulo
                            if re.match(pattern_negative, line_data[0]):
                                current_time = line_data[0][1:]
                            else:
                                # if is not a number the file must be deleted
                                if re.match(pattern_number, line_data[0]) is None:
                                    to_delete = 1
                                current_time = line_data[0]
                            # check sensor, if is in sensors_to_exclude don't consider
                            if line_data[1] not in const.SENSORS_TO_EXCLUDE_FROM_FILES:
                                current_sensor = line_data[1]
                                line_result = current_time + "," + current_sensor + "," + end_line
                                file_result.write(line_result)

                # remove files with incorrect values for time
                if to_delete == 1:
                    logging.info("  Delete: " + file + " --- Time with incorrect values")
                    deleted_files += 1
                    os.remove(res_file_path)

        # delete empty files
        file_empty = []
        filenames = listdir(const.DIR_RAW_DATA_CORRECT)
        for file in filenames:
            full_path = os.path.join(const.DIR_RAW_DATA_CORRECT, file)
            # check if file is empty
            if (os.path.getsize(full_path)) == 0:
                deleted_files += 1
                file_empty.append(file)
                logging.info("  Delete: " + file + " --- is Empty")
                os.remove(full_path)

        pattern = re.compile("^[0-9]+,[a-z,A-Z._]+,[-,0-9a-zA-Z.]+$", re.VERBOSE)
        # pattern = re.compile("^[0-9]+,[a-z,A-Z,\.,_]+,[-,0-9,a-z,A-Z,\.]+$", re.VERBOSE)
        filenames = listdir(const.DIR_RAW_DATA_CORRECT)
        for file in filenames:
            n_error = 0
            full_path = os.path.join(const.DIR_RAW_DATA_CORRECT, file)
            # check if all row respect regular expression
            with open(full_path) as f:
                for line in f:
                    match = re.match(pattern, line)
                    if match is None:
                        n_error += 1
            if n_error > 0:
                deleted_files += 1
                os.remove(full_path)

        logging.info("  Tot files in Dataset : " + str(n_files))
        logging.info("  Tot deleted files : " + str(deleted_files))
        logging.info("  Remaining files : " + str(len(listdir(const.DIR_RAW_DATA_CORRECT))))

        self.n_files = len(listdir(const.DIR_RAW_DATA_CORRECT))
        logging.info("END CLEAN FILES")
        print("END CLEAN.... results on log file")

    def transform_raw_data(self):
        """
        Transform sensor raw data in orientation independent data (with magnitude metric)
        :return:
        """
        dir_src = const.DIR_RAW_DATA_CORRECT
        dir_dst = const.DIR_RAW_DATA_TRANSFORM

        if not os.path.exists(dir_src):
            self.clean_files()

        self.create_or_reset_dir(dir_dst)

        if os.path.exists(dir_src):
            filenames = listdir(dir_src)
        else:
            shutil.rmtree(dir_dst)
            sys.exit("THERE ARE NO SYNTHETIC DATA TO BE PROCESSED")

        logging.info("TRANSFORMING RAW DATA...")
        print("TRANSFORMING RAW DATA...")
        for file in filenames:
            if file.endswith(".csv"):
                with open(os.path.join(dir_src, file)) as current_file:
                    with open(os.path.join(dir_dst, file), "w") as file_result:
                        for line in current_file:
                            line_data = line.split(",")
                            end_line = ",".join(line_data[2:])
                            current_time = line_data[0]
                            # sensor = line_data[1]
                            user = ""
                            target = ""
                            target = target.replace("\n", "")
                            # check sensors
                            if line_data[1] not in const.SENSORS_TO_EXCLUDE_FROM_DATASET:  # the sensor is not to exlude
                                if line_data[1] not in const.SENSOR_TO_TRANSFORM_MAGNITUDE:  # not to transofrom
                                    if line_data[1] not in const.SENSOR_TO_TRANSFROM_4ROTATION:  # not to trasform
                                        if line_data[1] not in const.SENSOR_TO_TAKE_FIRST:  # not to take first data
                                            # report the line as it is
                                            current_sensor = line_data[1]
                                            line_result = current_time + "," + current_sensor + "," + end_line
                                        else:
                                            current_sensor = line_data[1]
                                            vector_data = line_data[2:]
                                            try:
                                                vector_data = [float(i) for i in vector_data]
                                            except ValueError:
                                                vector_data = [str(i) for i in vector_data]
                                            line_result = current_time + "," + current_sensor + "," + str(
                                                vector_data[0]
                                            ) + user + target + "\n"
                                    else:  # the sensor is to transform 4 rotation
                                        current_sensor = line_data[1]
                                        vector_data = line_data[2:]
                                        vector_data = [float(i) for i in vector_data]
                                        magnitude = math.sin(math.acos(vector_data[3]))
                                        line_result = \
                                            current_time + "," + \
                                            current_sensor + "," + \
                                            str(magnitude) + user + target + "\n"
                                else:  # the sensor is to transform
                                    current_sensor = line_data[1]
                                    vector_data = line_data[2:]
                                    vector_data = [float(i) for i in vector_data]
                                    magnitude = math.sqrt(sum(((math.pow(vector_data[0], 2)),
                                                               (math.pow(vector_data[1], 2)),
                                                               (math.pow(vector_data[2], 2)))))
                                    line_result = \
                                        current_time + "," \
                                        + current_sensor + "," + \
                                        str(magnitude) + user + target + "\n"
                                file_result.write(line_result)
            elif file.endswith(".json"):
                shutil.copyfile(os.path.join(dir_src, file), os.path.join(dir_dst, file))
        logging.info("END TRANSFORMING RAW DATA...")
        print("END TRANSFORMING RAW DATA...")

    def fill_data_structure(self):
        """
        Fill travel modes, users, sensors data structures
        :return:
        """

        dir_src = const.DIR_RAW_DATA_TRANSFORM
        if not os.path.exists(dir_src):
            print("You should clean files first!")
            return -1

        filenames = listdir(dir_src)

        for file in filenames:
            if file.endswith(".csv"):
                data = file.split("_")
                if data[2] not in self.tm:
                    self.tm.append(data[2])
                if data[1] not in self.users:
                    self.users.append(data[1])
                f = open(os.path.join(dir_src, file))
                reader = csv.reader(f, delimiter=",")
                for row in reader:
                    if row[1] not in self.sensors and not row[1] == "":
                        self.sensors.append(row[1])
                f.close()

        self.header_with_features = {0: "time", 1: "activityrecognition#0", 2: "activityrecognition#1"}
        header_index = 3
        for s in self.sensors:
            if s != "activityrecognition":
                for feature_key in self.EXTRACTED_FEATURE_KEYS:
                    self.header_with_features[header_index] = s + feature_key
                    header_index += 1

        self.header = {0: "time", 1: "activityrecognition#0", 2: "activityrecognition#1"}
        header_index = 3
        for s in self.sensors:
            if s != "activityrecognition":
                self.header[header_index] = s + "#0"
                header_index += 1

    def range_position_in_header_with_features(self, sensor_name):
        """
        Return position of input sensor in header with features
        :param sensor_name:
        :return:
        """
        if len(self.header) == 0 or len(self.header_with_features) == 0:
            self.fill_data_structure()
        range_position = []
        start_pos = end_pos = -1
        i = 0
        found = False
        while True and i < len(self.header_with_features):
            compare = (str(self.header_with_features[i])).split("#")[0]
            if compare == sensor_name:
                found = True
                if start_pos == -1:
                    start_pos = i
                else:
                    end_pos = i
                i += 1
            else:
                i += 1
                if found:
                    if end_pos == -1:
                        end_pos = i - 2
                    break
        range_position.append(start_pos)
        range_position.append(end_pos)
        return range_position

    def range_position_in_header(self, sensor_name):
        """
        Return position of input sensor in header without features
        :param sensor_name:
        :return:
        """
        if len(self.header) == 0 or len(self.header_with_features) == 0:
            self.fill_data_structure()
        range_position = []
        start_pos = end_pos = -1
        i = 0
        found = False
        while True and i < len(self.header):
            compare = (str(self.header[i])).split("#")[0]
            if compare == sensor_name:
                found = True
                if start_pos == -1:
                    start_pos = i
                else:
                    end_pos = i
                i += 1
            else:
                i += 1
                if found:
                    if end_pos == -1:
                        end_pos = i - 2
                    break
        if end_pos == -1:
            end_pos = len(self.header) - 1
        range_position.append(start_pos)
        range_position.append(end_pos)
        return range_position

    def create_header_files(self):
        """
        Fill directory with all file consistent with the header without features
        :return:
        """

        dir_src = const.DIR_RAW_DATA_TRANSFORM
        dir_dst = const.DIR_RAW_DATA_HEADER

        if not os.path.exists(dir_src):
            print(dir_src)
            self.transform_raw_data()

        if len(self.header) == 0 or len(self.header_with_features) == 0:
            self.fill_data_structure()

        self.create_or_reset_dir(dir_dst)

        print("CREATE HEADER FILES....")
        filenames = listdir(dir_src)

        for file in filenames:
            if file.endswith(".csv"):
                user = ""
                target = ""
                current_file_data = file.split("_")
                target = current_file_data[2]
                user = current_file_data[1]
                full_current_file_path = os.path.join(dir_src, file)
                with open(full_current_file_path) as current_file:
                    full_current_file_path = os.path.join(dir_dst, file)
                    with open(full_current_file_path, "w") as file_header:
                        # write first line of file
                        header_line = ""
                        for x in range(0, len(self.header)):
                            if x == 0:  # time
                                header_line = self.header[0]
                            else:
                                header_line = header_line + "," + self.header[x]
                        header_line = header_line + ",target,user" + "\n"
                        file_header.write(header_line)
                        # write all other lines
                        j = -1
                        for line in current_file:
                            j += 1
                            line_data = line.split(",")
                            # first element time
                            new_line_data = {0: line_data[0]}
                            sensor_c = line_data[1]
                            pos = self.range_position_in_header(sensor_c)
                            # others elements all -1 except elements in range between pos[0] and pos[1]
                            curr_line_data = 2
                            for x in range(1, len(self.header)):  # x is the offset in list new_line_data
                                if x in range(pos[0], pos[1] + 1):
                                    if curr_line_data < len(line_data):
                                        if "\n" not in line_data[curr_line_data]:
                                            if "-Infinity" in line_data[curr_line_data]:
                                                new_line_data[x] = ""
                                            else:
                                                new_line_data[x] = line_data[curr_line_data]
                                        else:
                                            if "-Infinity" in line_data[curr_line_data]:
                                                new_line_data[x] = ""
                                            else:
                                                new_line_data[x] = line_data[curr_line_data].split("\n")[0]
                                        curr_line_data += 1
                                    else:
                                        new_line_data[x] = ""
                                else:
                                    new_line_data[x] = ""
                            new_line = ""
                            for x in range(0, len(new_line_data)):
                                if x == 0:
                                    new_line = new_line_data[0]
                                else:
                                    new_line = new_line + "," + new_line_data[x]
                            new_line = new_line + "," + str(target) + "," + str(user) + "\n"
                            file_header.write(new_line)
            elif file.endswith(".json"):
                shutil.copyfile(os.path.join(dir_src, file), os.path.join(dir_dst, file))
        print("END HEADER FILES....")

    def create_time_files(self):
        """
        Fill directory with all file consistent with the featured header divided in time window
        :return:
        """
        dir_src = const.DIR_RAW_DATA_HEADER
        dir_dst = const.DIR_RAW_DATA_FEATURES

        self.create_header_files_if_not_exist(dir_src)

        self.create_or_reset_dir(dir_dst)

        print("DIVIDE FILES IN TIME WINDOWS AND COMPUTE FEATURES....")
        # build string header
        header_string = ""
        for i in self.header_with_features:
            header_string = header_string + self.header_with_features[i] + ","
        header_string = header_string[:-1]
        header_string += ",target,user\n"

        # compute window dimension
        window_dim = int(const.SAMPLE_FOR_SECOND * const.WINDOW_DIMENSION)

        # loop on header files
        filenames = listdir(dir_src)
        current_values = []
        current_user = ""
        current_tm = ""
        for current_file in filenames:
            if current_file.endswith("csv"):

                current_tm = current_file.split("_")[2]
                current_user = current_file.split("_")[1]

                source_file_path = os.path.join(dir_src, current_file)
                df_file = pd.read_csv(source_file_path, dtype=const.DATASET_DATA_TYPE)

                feature_names = [
                    col for col in df_file.columns
                    if col not in [
                        'target',
                        'user',
                        'time',
                        'activityrecognition#0',
                        'activityrecognition#1'
                    ]
                ]

                # max time in source file
                end_time = df_file.loc[df_file['time'].idxmax()]['time']
                destination_file_path = os.path.join(dir_dst, current_file)
                destination_file = open(destination_file_path, 'w')
                destination_file.write(header_string)

                start_current = 0
                i = 0

                # track previuos value, if no value are present for a windows use previous
                list_of_feature_dicts = list()  # stored previous and current list of values

                for feature_key in self.EXTRACTED_FEATURE_KEYS:  # order is important!!!
                    list_of_feature_dicts.append({
                        self.FEATURE_KEY: feature_key,
                        self.PREVIOUS_LIST_KEY: list(),  # initialize all previous
                    })
                previous_activity_rec_list = ""
                previous_activity_rec_proba_list = ""
                previous_activity_rec = ""
                previous_activity_rec_proba = ""

                # loop on windows in file
                while True:

                    # clear all current
                    for feature_dict in list_of_feature_dicts:
                        feature_dict[self.CURRENT_LIST_KEY] = list()

                    # define time range
                    end_current = start_current + window_dim
                    if end_time <= end_current:
                        range_current = list(range(start_current, end_time, 1))
                        start_current = end_time
                    else:
                        range_current = list(range(start_current, end_current, 1))
                        start_current = end_current
                    # df of the current time window
                    df_current = df_file.loc[df_file['time'].isin(range_current)]
                    nfeature = 0

                    current_line = ""
                    for feature in feature_names:
                        current_feature_serie = df_current[feature]

                        dict_of_current_values = dict()  # features must be stored in the same order as headers

                        # time domain features
                        dict_of_current_values['#mean'] = current_feature_serie.mean(skipna=True)  # current mean
                        dict_of_current_values['#median'] = current_feature_serie.median(skipna=True)  # current median
                        dict_of_current_values['#min'] = current_feature_serie.min(skipna=True)
                        dict_of_current_values['#max'] = current_feature_serie.max(skipna=True)
                        dict_of_current_values['#std'] = current_feature_serie.std(skipna=True)
                        dict_of_current_values['#skewness'] = current_feature_serie.skew(skipna=True)
                        dict_of_current_values['#kurtosis'] = current_feature_serie.kurtosis(skipna=True)
                        dict_of_current_values['#ptp'] = current_feature_serie.ptp(skipna=True)  # max - min
                        current_q1 = current_feature_serie.quantile(0.25)  # 1st quantile
                        dict_of_current_values['#q1'] = current_q1
                        current_q3 = current_feature_serie.quantile(0.75)  # 3rd quantile
                        dict_of_current_values['#q3'] = current_q3
                        dict_of_current_values['#iqrange'] = current_q3 - current_q1  # interquantile range
                        dict_of_current_values['#var'] = current_feature_serie.var(skipna=True)  # variance

                        current_feature_serie.fillna(0, inplace=True)  # replace nan with 0 for next features
                        dict_of_current_values['#entropy'] = util.entropy(current_feature_serie.values)  # entropy

                        # frequency domain features
                        n_samples = len(current_feature_serie)

                        try:
                            fft = np.fft.rfft(current_feature_serie.values)
                            fft_frequency = np.fft.rfftfreq(n_samples)
                            fft_magnitude = np.abs(fft)
                            fft_phase = np.angle(fft)
                            fft_spectrum = np.abs(fft)**2
                            fft_highest_magnitude_index = np.argmax(fft_magnitude)
                            dict_of_current_values['#fft_highest_magnitude'] = fft_magnitude[fft_highest_magnitude_index]
                            dict_of_current_values['#fft_highest_magnitude_frequency'] = fft_frequency[
                                fft_highest_magnitude_index
                            ]
                            current_fft_total_spectrum = np.sum(fft_spectrum)  # total spectrum
                            dict_of_current_values['#fft_total_spectrum'] = current_fft_total_spectrum
                            spectral_densities = fft_spectrum / n_samples
                            current_fft_spectral_density = np.sum(spectral_densities)   # spectral density
                            dict_of_current_values['#fft_spectral_density'] = current_fft_spectral_density
                            normalized_spectral_densities = spectral_densities / current_fft_total_spectrum
                            dict_of_current_values['#fft_spectral_entropy'] = - np.sum(
                                normalized_spectral_densities * np.log(normalized_spectral_densities)
                            )  # spectral entropy

                            dict_of_current_values['#fft_spectral_centroid'] = (
                                np.sum(fft_magnitude*fft_frequency) / np.sum(fft_magnitude)  # spectral centroid
                            )
                            dict_of_current_values['#fft_total_phase'] = np.sum(fft_phase)  # total phase
                        except ValueError:
                            dict_of_current_values['#fft_highest_magnitude'] = 0.0
                            dict_of_current_values['#fft_highest_magnitude_frequency'] = 0.0
                            dict_of_current_values['#fft_total_spectrum'] = 0.0
                            dict_of_current_values['#fft_spectral_density'] = 0.0
                            dict_of_current_values['#fft_spectral_entropy'] = 0.0
                            dict_of_current_values['#fft_spectral_centroid'] = 0.0
                            dict_of_current_values['#fft_total_phase'] = 0.0


                        assert len(list_of_feature_dicts) == len(dict_of_current_values.keys())
                        if i == 0:
                            for feature_dict in list_of_feature_dicts:
                                feature_value = dict_of_current_values[feature_dict[self.FEATURE_KEY]]
                                feature_dict[self.PREVIOUS_LIST_KEY].append(str(feature_value))
                                feature_dict[self.CURRENT_LIST_KEY].append(str(feature_value))
                        else:
                            for feature_dict in list_of_feature_dicts:
                                feature_value = dict_of_current_values[feature_dict[self.FEATURE_KEY]]
                                if feature_value == 'nan':
                                    feature_dict[self.CURRENT_LIST_KEY].append(
                                        str(feature_dict[self.PREVIOUS_LIST_KEY][nfeature])
                                    )
                                else:
                                    feature_dict[self.CURRENT_LIST_KEY].append(str(feature_value))

                            for feature_dict in list_of_feature_dicts:
                                current_line = current_line + str(feature_dict[self.CURRENT_LIST_KEY][nfeature]) + ","

                            nfeature += 1

                    if df_current.shape[0] > 0:
                        # select 'activityrecognition#0' and 'activityrecognition#1' from  df_current
                        df_current_google = df_current[['activityrecognition#0', 'activityrecognition#1']]
                        df_current_google = df_current_google[df_current_google['activityrecognition#1'] >= 0]
                        if df_current_google.shape[0] == 0:
                            current_values.append(previous_activity_rec)
                            current_values.append(previous_activity_rec_proba)
                        else:
                            if df_current_google.shape[0] == 1:
                                df_row = df_current_google
                                current_values.append(df_row['activityrecognition#0'].item())
                                current_values.append(df_row['activityrecognition#1'].item())
                            else:
                                # pick prediction with max probability to be correct
                                activity0 = df_current_google.loc[df_current_google['activityrecognition#1'].idxmax()][
                                    'activityrecognition#0']
                                activity1 = df_current_google.loc[df_current_google['activityrecognition#1'].idxmax()][
                                    'activityrecognition#1']
                                current_values.append(activity0)
                                current_values.append(activity1)
                                previous_activity_rec = activity0
                                previous_activity_rec_proba = activity1

                    for feature_dict in list_of_feature_dicts:
                        feature_dict[self.PREVIOUS_LIST_KEY] = feature_dict[self.CURRENT_LIST_KEY]

                    if len(current_line) > 2:
                        line = str(i) + "," \
                               + str(current_values[0]) + "," \
                               + str(current_values[1]) + "," \
                               + current_line[:-1]
                        line = line + "," + str(current_tm) + "," + str(current_user) + "\n"
                        destination_file.write(line)
                    i += 1
                    if start_current == end_time:
                        break
        print("END DIVIDE FILES IN TIME WINDOWS AND COMPUTE FEATURES......")

    def create_or_reset_dir(self, dir_dst):
        if not os.path.exists(dir_dst):
            os.makedirs(dir_dst)
        else:
            shutil.rmtree(dir_dst)
            os.makedirs(dir_dst)

    @staticmethod
    def add_to_current_and_previous_list(current_value, current_value_list, previous_value_list):
        previous_value_list.append(str(current_value))
        current_value_list.append(str(current_value))

    def split_dataset(self, df):
        """
        Split passed dataframe into test, train and cv
        :param df:
        :return:
        """
        dir_src = const.DIR_DATASET
        file_training_dst = const.FILE_TRAINING
        file_test_dst = const.FILE_TEST
        file_cv_dst = const.FILE_CV

        training, cv, test = util.split_data(
            df,
            train_perc=const.TRAINING_PERC,
            cv_perc=const.CV_PERC,
            test_perc=const.TEST_PERC
        )
        training.to_csv(dir_src + '/' + file_training_dst, index=False)
        test.to_csv(dir_src + '/' + file_test_dst, index=False)
        cv.to_csv(dir_src + '/' + file_cv_dst, index=False)

    def preprocess_files(self):
        """
        Clean files and transform in orientation independent
        :return:
        """
        print("START PREPROCESSING...")
        self.clean_files()
        self.transform_raw_data()

    def analyze_sensors_support(self):
        """
        for each sensors analyze user support
        put support result in sensor_support.csv [sensor,nr_user,list_users,list_classes]
        :return:
        """
        if not os.path.exists(const.DIR_RAW_DATA_CORRECT):
            print("You should pre-processing files first!")
            return -1
        if len(self.users) == 0 or len(self.sensors) == 0 or len(self.tm) == 0:
            self.fill_data_structure()
        # build data frame for user support
        columns = ['sensor', 'nr_user', 'list_users', 'list_classes']
        index = list(range(len(self.sensors)))
        df_sensor_analysis = pd.DataFrame(index=index, columns=columns)
        df_sensor_analysis['sensor'] = self.sensors
        filenames = listdir(const.DIR_RAW_DATA_CORRECT)
        n_users = []
        users_list = []
        classes_list = []
        for s in self.sensors:
            class_list = []
            user_list = []
            for file in filenames:
                if file.endswith(".csv"):
                    data = file.split("_")
                    f = open(os.path.join(const.DIR_RAW_DATA_CORRECT, file))
                    if data[2] not in class_list:
                        class_list.append(data[2])
                    reader = csv.reader(f, delimiter=",")
                    for row in reader:
                        if row[1] == s and data[1] not in user_list:
                            user_list.append(data[1])
                    f.close()
            n_users.append(len(user_list))
            index = df_sensor_analysis[df_sensor_analysis['sensor'] == s].index.tolist()
            df_sensor_analysis.ix[index, 'list_users'] = str(user_list)
            df_sensor_analysis.ix[index, 'list_classes'] = str(class_list)
            self.add_to_current_and_previous_list(class_list, classes_list, users_list)
        df_sensor_analysis['nr_user'] = n_users
        df_sensor_analysis['list_users'] = users_list
        df_sensor_analysis['list_classes'] = classes_list

        df_sensor_analysis = df_sensor_analysis.sort_values(by=['nr_user'], ascending=[False])

        # remove result file if exists
        try:
            os.remove(const.FILE_SUPPORT)
        except OSError:
            pass
        df_sensor_analysis.to_csv(const.FILE_SUPPORT, index=False)

    def get_remained_sensors(self, sensors_set):
        """
        Return list of considered sensors based on the correspondent classification level
        :param sensors_set:
        :return:
        """
        excluded_sensors = self.get_excluded_sensors(sensors_set)
        remained_sensors = []
        for s in self.get_sensors:
            if s not in excluded_sensors:
                s = s.replace('android.sensor.', '')
                remained_sensors.append(s)
        return remained_sensors

    def get_sensors_set_features(self, sensors_set):
        """
        Get set of features selected in sensor set
        :param sensors_set:
        :return:
        """
        feature_to_delete = []
        header = self.get_header
        for s in self.get_excluded_sensors(sensors_set):
            for x in header.values():
                if s in x:
                    feature_to_delete.append(x)
        features_list = (set(header.values()) - set(feature_to_delete))
        return features_list

    def get_sensor_features(self, sensor):
        """
        Get features by sensor
        :param sensor:
        :return:
        """
        feature_sensor = []
        header = self.get_header
        for x in header.values():
            if sensor in x:
                feature_sensor.append(x)
        return feature_sensor

    def get_feature_columns(self, dataset_type=const.DATASET_TYPE, **kwargs):
        if 'raw_signal' in dataset_type:
            feature_columns = self.get_best_raw_signals(**kwargs)
        else:
            feature_columns = self.get_best_window_features(**kwargs)
        return feature_columns

    def get_best_window_features(self, **kwargs):
        """
        :param kwargs
        :return:
        """

        # check if best features have been selected before
        best_features_filepath = const.DIR_DATASET + "/" + self.BEST_FEATURES_OUTPUT_FILE
        if os.path.isfile(best_features_filepath) is False:

            # load preprocessed dataset
            dataframe = self.get_preprocessed_dataset()
            all_features_columns = list(self.get_remained_sensors(0))
            features_dataframe = dataframe[all_features_columns]
            classes_dataframe = dataframe[self.TRAVEL_MODE_COLUMN]

            # execute feature selection with RFE
            feature_columns = self._get_best_sensor_features_with_rfe(
                features_dataframe=features_dataframe,
                classes_dataframe=classes_dataframe,
                **kwargs
            )
            features_dataframe = dataframe[feature_columns]
            n_features = len(feature_columns)

            # execute dimensionality reduction with KBest
            max_features = int(math.sqrt(len(dataframe)))  # rule of thumb
            feature_columns = self._get_best_sensor_features_with_kbest(
                features_dataframe=features_dataframe,
                classes_dataframe=classes_dataframe,
                k_features=min(n_features, max_features),
                **kwargs
            )

            # save selected features names in file
            with open(best_features_filepath, mode='x', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(['feature_header_column'])
                for column in feature_columns:
                    csvwriter.writerow([column])

        # retrieve best features lists
        return pd.read_csv(best_features_filepath).values.flatten().tolist()

    @staticmethod
    def _get_best_sensor_features_with_kbest(
            features_dataframe: pd.DataFrame,
            classes_dataframe: pd.DataFrame,
            k_features=20,
            metric=mutual_info_classif
    ):
        """
        select k best features with mututal_info_classif
        :param features_dataframe:
        :param classes_dataframe:
        :param k_features:
        :return:
        """
        feature_selector = SelectKBest(metric, k=k_features)
        feature_selector.fit(features_dataframe.values, classes_dataframe.values)
        feature_columns = features_dataframe.columns[feature_selector.get_support()]
        return feature_columns

    @staticmethod
    def _get_best_sensor_features_with_rfe(
            features_dataframe: pd.DataFrame,
            classes_dataframe: pd.DataFrame,
            estimator=RandomForestClassifier(),
            n_steps=1,
            n_folds=10,
            scoring_metric='accuracy',
            verbose=1
    ):
        """
        Retrieves the best sensor features selected through
        10-fold Cross Validated Recursive Feature Elimination
        with RandomForest estimator and accuracy metric
        :return:
        """
        feature_selector = RFECV(
            estimator=estimator,
            step=n_steps,
            cv=StratifiedKFold(n_folds),
            scoring=scoring_metric,
            verbose=verbose
        )
        feature_selector.fit(features_dataframe.values, classes_dataframe.values)
        feature_columns = features_dataframe.columns[feature_selector.get_support()]
        return feature_columns

    def get_preprocessed_dataset(self, dataset_type=const.DATASET_TYPE):
        """
        Retrieves the preprocessed dataset
        :param dataset_type ('feature_unbalanced', 'feature_balanced', 'raw_signal_forward', 'raw_signal_zero' )
        :return: dataframe:
        """
        print("Preprocessing dataset ...")

        if dataset_type == 'feature_unbalanced':
            dataframe = self.get_dataset
        elif dataset_type == 'feature_balanced':
            dataframe = self.get_balanced_dataset
        elif dataset_type == 'raw_signal_forward':
            dataframe = self.get_raw_dataset_with_forward_filling
        elif dataset_type == 'raw_signal_zero':
            dataframe = self.get_raw_dataset_with_zero_filling
        else:
            raise ValueError('dataset_type {} not supported!'.format(dataset_type))

        # dataframe = dataframe[dataframe.user != 'U1']  # remove user 1 to reduce bias
        dataframe = dataframe.fillna(dataframe.mean())  # fill na with mean
        dataframe = dataframe.fillna(0)  # fill na with zero
        self.scale_features(dataframe, dataset_type)  # scale features to reduce bias
        print('Finished preprocessing dataset!')
        return dataframe

    def scale_features(self, dataframe, dataset_type=const.DATASET_TYPE):
        """
        Scales dataframe features in place using interquartile range
        :param dataframe:
        :param dataset_type:
        :return:
        """
        if 'raw_signal' in dataset_type:
            feature_columns = self.get_raw_signals()
        else:
            feature_columns = list(self.get_sensors_set_features(0))
        print(feature_columns)
        features_dataframe = dataframe[feature_columns]
        dataframe[feature_columns] = RobustScaler().fit_transform(features_dataframe)

    @staticmethod
    def get_excluded_sensors(sensors_set):
        """
        Return list of excluded sensor based on the correspondent classification level
        :param sensors_set:
        :return:
        """
        if sensors_set == 1:
            excluded_sensors = const.sensor_to_exclude_first
        elif sensors_set == 2:
            excluded_sensors = const.sensor_to_exclude_second
        elif sensors_set == 3:
            excluded_sensors = const.sensors_to_exclude_third
        else:
            excluded_sensors = const.sensors_to_exclude_all
        return excluded_sensors

    # --------------------------------- PROPERTIES ------------------------------------------------------------------- #
    @property
    def get_users(self):
        if len(self.users) == 0:
            self.fill_data_structure()
        return self.users

    @property
    def get_tm(self):
        if len(self.tm) == 0:
            self.fill_data_structure()
        return self.tm

    @property
    def get_sensors(self):
        if len(self.sensors) == 0:
            self.fill_data_structure()
        return self.sensors

    @property
    def get_header(self):
        if len(self.header_with_features) == 0:
            self.fill_data_structure()
        return self.header_with_features

    @property
    def get_train(self):
        return pd.read_csv(const.DIR_DATASET + "/" + const.FILE_TRAINING)

    @property
    def get_test(self):
        return pd.read_csv(const.DIR_DATASET + "/" + const.FILE_TEST)

    @property
    def get_cv(self):
        return pd.read_csv(const.DIR_DATASET + "/" + const.FILE_CV)

    @property
    def get_dataset(self):
        return pd.read_csv(const.DIR_DATASET + "/" + const.FILE_DATASET)

    @property
    def get_raw_dataset_with_forward_filling(self):
        return pd.read_csv(const.DIR_RAW_DATASET + "/" + const.FILE_RAW_DATASET_FORWARD)

    @property
    def get_raw_dataset_with_zero_filling(self):
        return pd.read_csv(const.DIR_RAW_DATASET + "/" + const.FILE_RAW_DATASET_ZERO)

    @property
    def get_balanced_dataset(self):
        return pd.read_csv(const.DIR_DATASET + "/" + const.FILE_DATASET_BALANCED)


# Define Flower client
class TMDClient(fl.client.NumPyClient):

    def get_parameters(self):
        """Get parameters of the local model."""
        return model.get_weights()

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        model.set_weights(parameters)

        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)

        return model.get_weights(), len(x_train), {}

    # test the local model
    def evaluate(self, parameters, config):
        model.set_weights(parameters)

        loss, accuracy = model.evaluate(x_test, y_test)

        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client("localhost:8080", client=TMDClient())


if __name__ == "__main__":
    # create folders for dataset download
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if not os.path.exists(datasetBalanced):
        os.makedirs(datasetBalanced)

    if not os.path.exists(rawOriginaldata):
        os.makedirs(rawOriginaldata)

    print("DOWNLOAD........")
    for url in url_list:
        response = urllib2.urlopen(url)
        csv = response.read()
        if url=='http://cs.unibo.it/projects/us-tm2017/static/dataset/extension/5second/dataset_5secondWindow.csv':
            outfile = datasetBalanced + '/' +dataset5second
           elif url=='http://cs.unibo.it/projects/us-tm2017/static/dataset/raw_data/raw_data.tar.gz':
	    outfile = rawOriginaldata + '/' + rawdataorig
        else:
            outfile = datasetBalanced + '/' + datasethalfsecond

        with open(outfile, 'wb') as f:
            f.write(csv)

	if url == "http://cs.unibo.it/projects/us-tm2017/static/dataset/raw_data/raw_data.tar.gz":
	    tar = tarfile.open(outfile, "r:gz")
    	    tar.extractall(path="TransportationData/")
	    tar.close()
	    for filename in glob.iglob('TransportationData/raw_data/*/*.csv'):
		shutil.move(filename, rawOriginaldata)
	    os.remove(outfile)
	    shutil.rmtree('TransportationData/raw_data/')

    print "DOWNLOAD ENDED."

    # load dataset
    df = get_tmd_dataset()
    travel_mode_column = 'target'

    # Creating raw dataset
    dataset = TMDataset()
    dataset.create_raw_dataset()
