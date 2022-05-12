import pandas
import numpy
import os
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


class TravelModeDetector(object):
    """
    Wrapper class that encapsulates machine learning model
    implementations for travel mode detection
    """

    DEFAULT_MODEL_PATH = 'tmd_model'
    MODEL_FILENAME = 'model.pkl'
    CLASSES_FILENAME = 'classes.pkl'

    def __init__(self, save_path=DEFAULT_MODEL_PATH, **kwargs):
        """
        Initializes model
        """
        self.save_path = save_path
        self.model = None
        self.model_byte_size = None
        self.classes = []
        self.classes2string = {}
        self.classes2number = {}

    @property
    def get_model_byte_size(self):
        if self.model_byte_size is None:
            raise Exception('Model size has not been set yet!')
        return self.model_byte_size

    def fit(
            self,
            data_frame: pandas.DataFrame,
            travel_mode_column: str,
            **kwargs
    ):
        """
        Fit TMDModel to classify travel_mode_column based on data_columns.
        """
        raise NotImplementedError

    def predict(self, data_frame: pandas.DataFrame, **kwargs):
        """
        Detects the travel mode of each batch_size in data_frame.
        """
        self.check_if_model_is_trained()
        data_frame = self.fill_nan_with_mean(data_frame)
        predictions = self.model.predict(data_frame.values)
        predictions = self.convert_numbers_to_classes(pandas.DataFrame(predictions))
        return predictions

    def save(self, model_path: str = DEFAULT_MODEL_PATH, **kwargs):
        """
        Saves already trained model to file
        :param model_path:
        :param kwargs:
        :return:
        """
        self.check_if_model_is_trained()
        self.create_model_path(model_path)
        model_file_name = os.path.join(model_path, self.MODEL_FILENAME)
        joblib.dump(self.model, model_file_name)
        self.model_byte_size = os.path.getsize(model_file_name)
        joblib.dump(self.classes2string, os.path.join(model_path, self.CLASSES_FILENAME))

    def restore(self, model_path: str = DEFAULT_MODEL_PATH):
        """
        Loads trained model from file
        :param model_path:
        :return:
        """
        self.model = joblib.load(os.path.join(model_path, self.MODEL_FILENAME))
        self.classes2string = joblib.load(os.path.join(model_path, self.CLASSES_FILENAME))

    def check_if_model_is_trained(self):
        """
        :return:
        """
        assert self.model is not None, 'Model must be trained before this action!'

    def get_preprocessed_partitions(
            self,
            data_frame,
            travel_mode_column,
            test_ratio=0.1,
            train_ratio=0.8,
            val_ratio=0.1,
            shuffle=True,
            convert_modes_to_numbers=False,
            fill_nan_with_mean=True,
            one_hot_encode=False,
            convert_to_numpy_array=True,
            standardize=False,
            normalize_to_z_scale=False
    ):
        """
        Get preprocessed train, validation and test partitions for x and y
        :param data_frame:
        :param travel_mode_column:
        :param test_ratio:
        :param train_ratio:
        :param val_ratio:
        :param shuffle:
        :param convert_modes_to_numbers
        :param fill_nan_with_mean
        :param one_hot_encode
        :param convert_to_numpy_array
        :param standardize
        :param normalize_to_z_scale
        :return:
        """

        # shuffle dataframe
        if shuffle:
            data_frame = data_frame.sample(frac=1)

        # separate input from output
        df_x = data_frame.copy()
        del df_x[travel_mode_column]
        df_y = data_frame[travel_mode_column]

        # preprocess
        if fill_nan_with_mean:
            df_x = TravelModeDetector.fill_nan_with_mean(df_x)
        if convert_modes_to_numbers:
            df_y = self.convert_classes_to_numbers(df_y)
        if standardize:  # Stardardize to gaussian distribution with zero mean and unit variance
            df_x = self.standardize(df_x)
        if normalize_to_z_scale:
            df_x = self.normalize_to_z_scale(df_x)

        # get partitions
        x_test, x_train, x_val, y_test, y_train, y_val = TravelModeDetector.get_partitions(
            df_x, df_y, test_ratio, train_ratio, val_ratio, convert_to_numpy_array
        )

        # encode
        if one_hot_encode:
            n_classes = data_frame[travel_mode_column].nunique()
            y_train, train_labels = TravelModeDetector.factorize_and_one_hot_encode(n_classes, y_train)
            y_val, val_labels = TravelModeDetector.factorize_and_one_hot_encode(n_classes, y_val)
            y_test, test_labels = TravelModeDetector.factorize_and_one_hot_encode(n_classes, y_test)
            self.classes = train_labels

        return x_test, x_train, x_val, y_test, y_train, y_val

    @staticmethod
    def standardize(df_x: pandas.DataFrame):
        """
        :param df_x:
        :return:
        """
        df_x = pandas.DataFrame(preprocessing.scale(df_x.values), columns=df_x.columns)
        return df_x

    @staticmethod
    def normalize_to_z_scale(df_x: pandas.DataFrame):
        """
        :param df_x:
        :return:
        """
        z_scaler = StandardScaler()
        df_x = pandas.DataFrame(z_scaler.fit_transform(df_x.values), columns=df_x.columns)
        return df_x

    @staticmethod
    def factorize_and_one_hot_encode(n_classes, y):
        """
        :param n_classes:
        :param y:
        :return:
        """
        y = numpy.sort(y)
        y, labels = pandas.factorize(y)
        y = TravelModeDetector.one_hot_encode(y, n_classes=n_classes)
        return y, labels

    @staticmethod
    def get_partitions(
            df_x,
            df_y,
            test_ratio=0.1,
            train_ratio=0.8,
            val_ratio=0.1,
            convert_to_numpy_array=True,
            batch_size_for_truncation=None
    ):
        """
        Get train, validation and test partitions for x and y
        :param df_x:
        :param df_y:
        :param test_ratio:
        :param train_ratio:
        :param val_ratio:
        :param convert_to_numpy_array:
        :param batch_size_for_truncation
        :return:
        """
        assert (train_ratio + val_ratio + test_ratio) == 1.0

        n_rows = len(df_x)
        n_train = int(n_rows * train_ratio)
        n_val = int(n_rows * val_ratio)

        x_train = df_x[:n_train]
        y_train = df_y[:n_train]

        x_val = df_x[n_train:n_train + n_val]
        y_val = df_y[n_train:n_train + n_val]

        x_test = df_x[n_train + n_val:]
        y_test = df_y[n_train + n_val:]

        # drop samples to make number of train, val and test samples a multiple
        # of the batch size
        if batch_size_for_truncation:
            x_train, y_train = TravelModeDetector.truncate_samples(batch_size_for_truncation, x_train, y_train)
            x_val, y_val = TravelModeDetector.truncate_samples(batch_size_for_truncation, x_val, y_val)
            x_test, y_test = TravelModeDetector.truncate_samples(batch_size_for_truncation, x_test, y_test)

        if convert_to_numpy_array:
            x_train = x_train.values
            x_val = x_val.values
            x_test = x_test.values
            y_train = y_train.values
            y_val = y_val.values
            y_test = y_test.values

        return x_test, x_train, x_val, y_test, y_train, y_val

    def convert_classes_to_numbers(self, df_y: pandas.DataFrame):
        """
        Convert classes to numbers
        :param df_y:
        :return:
        """
        for i, c in enumerate(sorted(set(df_y))):
            self.classes2string[i] = c
            self.classes2number[c] = i
            self.classes.append(c)
        df_y_list = [self.classes2number[c] for c in df_y.values]
        return pandas.DataFrame(numpy.array(df_y_list))

    def convert_numbers_to_classes(self, df_y: pandas.DataFrame):
        """
        Convert numbers to classes
        :param df_y:
        :return:
        """
        assert len(self.classes2string) > 0
        df_y_list = [self.classes2string[int(n[0])] for n in df_y.values]
        return pandas.Series(numpy.array(df_y_list))

    @staticmethod
    def fill_nan_with_mean(dataframe: pandas.DataFrame):
        """
        Fill nan values in a dataframe with mean value and 0
        :param dataframe:
        :return:
        """
        dataframe_fill = dataframe.copy()
        dataframe_fill = dataframe_fill.fillna(dataframe_fill.mean())
        dataframe_fill = dataframe_fill.fillna(0)
        return dataframe_fill

    @staticmethod
    def truncate_samples(batch_size, x, y):
        """
        Truncates samples to generate only full batches
        :param batch_size:
        :param x:
        :param y:
        :return:
        """
        n_samples = x.shape[0]
        n_batches = n_samples / batch_size
        n_full_batches = int(n_batches)
        n_samples = n_full_batches * batch_size
        return x[:n_samples], y[:n_samples]

    @staticmethod
    def get_percentage_missing(series: pandas.Series):
        """
        Calculates percentage of NaN values in DataFrame
        :param series: Pandas DataFrame object
        :return: float
        """
        num = series.isnull().sum()
        den = len(series)
        return round(num / den, 2)

    @staticmethod
    def create_model_path(model_path):
        """
        :param model_path:
        :return:
        """
        if os.path.isdir(model_path) is False:
            os.mkdir(model_path)

    @staticmethod
    def print_var_hashes(sess):
        import hashlib
        import tensorflow as tf
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            val = sess.run(v.value())
            h = hashlib.md5()
            h.update(val.data)
            print('%-50s %s' % (v.name, h.hexdigest()))

    @staticmethod
    def one_hot_encode(y, n_classes=None):
        """
        Convert `y` to one-hot encoding.
        :param y:
        :param n_classes:
        :return:
        """
        n_classes = n_classes or numpy.max(y) + 1
        return numpy.eye(n_classes)[y]
