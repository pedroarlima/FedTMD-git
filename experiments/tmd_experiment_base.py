from detectors.tmd_base import TravelModeDetector
from datasets.tmdataset import TMDataset
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold
import pandas as pd
from datetime import datetime
import sklearn.metrics
from os import path, mkdir
from imblearn.over_sampling import SMOTE
import csv
import os


class TMDExperiment(object):
    """
    Object that implements methods required for running an evaluation experiment of a TMDdetector
    """

    CONFIGURATIONS_INPUT_FILE = 'configurations.csv'
    METRICS_OUTPUT_FILE = 'metrics.csv'
    TEST_OUTPUT_FILE = 'test.csv'
    TRAIN_OUTPUT_FILE = 'train.csv'
    OUTPUT_DIR = 'experiment_runs'

    def __init__(
            self,
            experiment_path,
            detector_type=TravelModeDetector,
            n_repeats=1,
            n_folds=10
    ):
        """
        :param experiment_path:
        :param n_repeats:
        :param n_folds:
        :param detector_type:
        """
        self.experiment_path = experiment_path
        self.tmdataset = TMDataset()
        self.n_repeats = n_repeats
        self.n_folds = n_folds
        self.detector_type = detector_type

    def run(self):
        """
        Execute experiment
        :return:
        """

        # get dataset and detector configurations
        dataframe = self.tmdataset.get_preprocessed_dataset()
        travel_mode_column = self.tmdataset.TRAVEL_MODE_COLUMN
        classes_dataframe = dataframe[travel_mode_column]
        feature_columns = self.tmdataset.get_feature_columns(include_indicators=os.getenv('INCLUDE_INDICATORS', True))
        features_dataframe = dataframe[feature_columns]
        user_id_column = self.tmdataset.USER_ID_COLUMN
        users_dataframe = dataframe[user_id_column]
        list_of_detector_configurations = self.get_detector_configurations()

        # create output dir
        output_path = path.join(self.OUTPUT_DIR)
        if path.isdir(output_path) is False:
            mkdir(output_path)

        # create metrics file
        metrics_filepath = path.join(output_path, self.METRICS_OUTPUT_FILE)
        if path.isfile(metrics_filepath) is False:
            with open(metrics_filepath, mode='x', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow([
                    'configuration_id', 'repetition', 'fold',
                    'train_time', 'detector_byte_size', 'train_prediction_time', 'test_prediction_time',
                    'train_accuracy', 'train_precision', 'train_recall', 'train_f1_score', 'train_cohen_kappa_score',
                    'test_accuracy', 'test_precision', 'test_recall', 'test_f1_score', 'test_cohen_kappa_score'
                ])

        # run cross validation for each sensor set, for each configuration
        for detector_configuration in list_of_detector_configurations:

            configuration_id = detector_configuration.pop('configuration_id')

            print('Running experiments with configuration {} ...'.format(configuration_id))

            # create outputs dir
            configuration_path = path.join(output_path, configuration_id)
            if path.isdir(configuration_path) is False:
                mkdir(configuration_path)
            #try:
            self.evaluate_with_cross_validation(
                classes_dataframe, configuration_id, feature_columns,
                travel_mode_column, features_dataframe, metrics_filepath,
                detector_configuration, configuration_path, users_dataframe
            )
            #except Exception as e:
            #    print("Configuration {} failed with error {}".format(configuration_id, str(e)))

    # ------------------------------------------------- SUBROUTINES -------------------------------------------------- #

    def evaluate_with_cross_validation(
            self,
            classes_dataframe,
            configuration_id,
            feature_columns,
            travel_mode_column,
            features_dataframe,
            metrics_filepath,
            detector_configuration,
            configuration_path,
            users_dataframe,
    ):
        """
        :param classes_dataframe:
        :param configuration_id:
        :param feature_columns:
        :param travel_mode_column:
        :param features_dataframe:
        :param metrics_filepath:
        :param detector_configuration:
        :param configuration_path:
        :param users_dataframe:
        :return:
        """

        # get cross validation datasets
        # skf = GroupKFold(n_splits=self.n_folds)
        skf = RepeatedStratifiedKFold(n_splits=self.n_folds, n_repeats=self.n_repeats)

        # train and evaluate detector for each fold and repetition
        current_fold = 1
        current_repetition = 1
        for train_indices, test_indices in skf.split(X=features_dataframe, y=classes_dataframe):

            print('Evaluating fold {f} of repetition {r} ...'.format(f=current_fold, r=current_repetition))

            # check if fold has already been evaluated
            repetition_path = path.join(
                configuration_path,
                'repetition_{}'.format(current_repetition),
            )
            if path.isdir(repetition_path) is False:
                mkdir(repetition_path)
            fold_path = path.join(
                repetition_path,
                'fold_{}'.format(current_fold)
            )
            if path.isdir(fold_path) is False:
                mkdir(fold_path)
            train_output_path = path.join(fold_path, self.TRAIN_OUTPUT_FILE)
            test_output_path = path.join(fold_path, self.TEST_OUTPUT_FILE)
            if path.isfile(train_output_path) is False or path.isfile(test_output_path) is False:

                # get train features and classes
                train_features = features_dataframe.iloc[train_indices]
                train_classes = classes_dataframe.iloc[train_indices]

                print("Resampling train dataset with SMOTE ...")
                smote = SMOTE(ratio='all')
                resampled_train_features, resampled_train_classes = smote.fit_sample(train_features, train_classes)
                print("Finished resampling with SMOTE!")

                # concatenate in train set
                resampled_train_features = pd.DataFrame(resampled_train_features, columns=feature_columns)
                resampled_train_classes = pd.DataFrame(resampled_train_classes, columns=[travel_mode_column])
                train_dataframe = pd.concat(
                    [resampled_train_features, resampled_train_classes],
                    axis=1,
                    sort=False
                )

                # get test set
                test_features = features_dataframe.iloc[test_indices]
                test_classes = classes_dataframe.iloc[test_indices]

                travel_mode_detector = self.get_travel_mode_detector(fold_path)

                print("Training Detector ...")
                train_start_time = datetime.now()
                self.fit_travel_mode_detector(
                    detector_configuration,
                    travel_mode_detector,
                    train_dataframe,
                    travel_mode_column,
                )
                train_time = datetime.now() - train_start_time
                print("Finished Training!")

                # extract detector size
                detector_byte_size = travel_mode_detector.model_byte_size

                # evaluate on training and test sets
                batch_size = detector_configuration.get('batch_size', None)
                print("Evaluating Detector on Training Set ...")

                train_accuracy, train_cohen_kappa_score, train_f1_score, train_precision, train_prediction_time, \
                    train_recall = self.evaluate_detector(
                        batch_size=batch_size,
                        travel_mode_detector=travel_mode_detector,
                        classes=train_classes,
                        features=train_features,
                        output_file_path=train_output_path
                    )
                print("Finished Evaluating!")
                print("Evaluating detector on Test Set ...")
                test_accuracy, test_cohen_kappa_score, test_f1_score, test_precision, test_prediction_time, \
                    test_recall = self.evaluate_detector(
                        batch_size=batch_size,
                        travel_mode_detector=travel_mode_detector,
                        classes=test_classes,
                        features=test_features,
                        output_file_path=test_output_path
                    )
                print("Finished Evaluating!")

                print("Storing Fold Results ...")
                # store metrics
                with open(metrics_filepath, mode='a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',')
                    csvwriter.writerow([
                        configuration_id, current_repetition, current_fold,
                        train_time, detector_byte_size, train_prediction_time, test_prediction_time,
                        train_accuracy, train_precision, train_recall, train_f1_score, train_cohen_kappa_score,
                        test_accuracy, test_precision, test_recall, test_f1_score, test_cohen_kappa_score,
                    ])
                print("Finished Storing!")
            else:
                print("Fold {} already evaluated!".format(current_fold))

            # increment fold and repetition counters
            current_fold += 1
            if current_fold > self.n_folds:
                current_fold = 1
                current_repetition += 1

    def get_travel_mode_detector(self, fold_path):
        """
        Retrieves Travel Mode Detector Instance
        :param fold_path:
        :return:
        """
        travel_mode_detector = self.detector_type(save_path=fold_path)
        return travel_mode_detector

    def fit_travel_mode_detector(
            self,
            detector_configuration,
            travel_mode_detector,
            train_dataframe,
            travel_mode_column,
            **kwargs
    ):
        """
        Trains travel mode detector instance
        :param detector_configuration:
        :param travel_mode_detector:
        :param train_dataframe:
        :param travel_mode_column
        :param kwargs:
        :return:
        """
        travel_mode_detector.fit(
            data_frame=train_dataframe.copy(),
            travel_mode_column=travel_mode_column,
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0,
            **detector_configuration
        )

    def get_detector_configurations(self):
        """
        Read configurations from file
        :return:
        """
        print('Reading detector configurations ...')
        list_of_detector_configurations = list()
        configuration_filepath = path.join(self.experiment_path, self.CONFIGURATIONS_INPUT_FILE)
        configuration_reader = csv.DictReader(open(configuration_filepath, 'r'))
        for detector_configuration in configuration_reader:
            for key in detector_configuration.keys():
                detector_configuration[key] = eval(detector_configuration[key])
            list_of_detector_configurations.append(detector_configuration)
        print('Finished reading configurations!')
        return list_of_detector_configurations

    def evaluate_detector(self, batch_size, travel_mode_detector, classes, features, output_file_path):
        """
        Evaluates a tmd detector with a given input and output and stores the results in a CSV
        :param batch_size:
        :param travel_mode_detector:
        :param classes:
        :param features:
        :param output_file_path
        :return:
        """
        predictions, prediction_time = self.get_predictions(
            batch_size=batch_size,
            tmd_detector=travel_mode_detector,
            input_features=features
        )
        accuracy, cohen_kappa_score, f1_score, precision, recall = self.get_classification_metrics(
            classes=classes,
            predictions=predictions
        )
        self.store_classes_and_predictions(output_file_path, classes, predictions)
        return accuracy, cohen_kappa_score, f1_score, precision, prediction_time, recall

    @staticmethod
    def store_classes_and_predictions(output_file_path, classes, predictions):
        """
        Writes classes and predictions in a CSV file
        :param output_file_path:
        :param classes:
        :param predictions:
        :return:
        """
        with open(output_file_path, mode='a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(['true', 'predicted'])
            for i in range(len(classes)):
                csvwriter.writerow([classes.iloc[i], predictions.iloc[i]])

    @staticmethod
    def get_classification_metrics(classes, predictions):
        """
        Retrieves classification metrics
        :param classes:
        :param predictions:
        :return: accuracy, cohen_kappa_score, f1_score, precision, recall
        """
        accuracy = sklearn.metrics.accuracy_score(classes, predictions)
        precision = sklearn.metrics.precision_score(classes, predictions, average='macro')
        recall = sklearn.metrics.recall_score(classes, predictions, average='macro')
        f1_score = sklearn.metrics.f1_score(classes, predictions, average='macro')
        cohen_kappa_score = sklearn.metrics.cohen_kappa_score(classes, predictions)
        return accuracy, cohen_kappa_score, f1_score, precision, recall

    @staticmethod
    def get_predictions(batch_size, tmd_detector, input_features):
        """
        Get predictions using an already trained detector
        :param batch_size:
        :param tmd_detector:
        :param input_features:
        :return: predictions, prediction_time
        """
        prediction_start_time = datetime.now()
        predictions = tmd_detector.predict(
            data_frame=input_features,
            batch_size=batch_size,
            verbose=0
        )
        prediction_time = datetime.now() - prediction_start_time
        return predictions, prediction_time
