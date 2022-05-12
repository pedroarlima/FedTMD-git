import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets.tmdataset import TMDataset
from sklearn.metrics import confusion_matrix
import os
from experiments.tmd_experiment_base import TMDExperiment


class TMDExperimentAnalysis(object):

    ANALYSIS_PATH = 'analysis'
    EXPERIMENTS_PATH = 'experiments'
    EXPERIMENT_TEST_OUTPUT_FILE = TMDExperiment.TEST_OUTPUT_FILE
    EXPERIMENT_TRAIN_OUTPUT_FILE = TMDExperiment.TRAIN_OUTPUT_FILE
    EXPERIMENT_METRICS_OUTPUT_FILE = TMDExperiment.METRICS_OUTPUT_FILE
    EXPERIMENT_OUTPUT_DIR = TMDExperiment.OUTPUT_DIR
    CONFUSION_MATRIX_TEST_OUTPUT_SUFFIX = 'test.svg'
    CONFUSION_MATRIX_TRAIN_OUTPUT_SUFFIX = 'train.svg'
    CONFUSION_MATRIX_OUTPUT_DIR = os.path.join(ANALYSIS_PATH, 'confusion_matrices')
    METRIC_TABLES_OUTPUT_DIR = os.path.join(ANALYSIS_PATH, 'metric_tables')

    def run(self):

        # get classes
        tmddataset = TMDataset()
        dataframe = tmddataset.get_preprocessed_dataset()
        modes = dataframe.target.unique()

        # create confusion matrix output dir
        self.check_or_create_dir(self.CONFUSION_MATRIX_OUTPUT_DIR)
        self.check_or_create_dir(self.METRIC_TABLES_OUTPUT_DIR)

        # list all experiment folders
        experiment_dirs = os.listdir(self.EXPERIMENTS_PATH)
        # for each experiment folder
        for experiment_dir in experiment_dirs:
            if experiment_dir.endswith("tmd") and os.path.isdir(os.path.join(self.EXPERIMENTS_PATH, experiment_dir)):
                print("Analyzing experiment {}".format(experiment_dir))
                self.generate_metrics_table(experiment_dir)
                self.generate_confusion_matrices_for_each_configuration(experiment_dir, modes)

    def generate_metrics_table(self, experiment_dir):

        # get metrics csv
        experiment_result_dir_path = os.path.join(self.EXPERIMENTS_PATH, experiment_dir, self.EXPERIMENT_OUTPUT_DIR)
        metrics_output = pd.read_csv(os.path.join(experiment_result_dir_path, self.EXPERIMENT_METRICS_OUTPUT_FILE))

        # average metrics by configuration_id
        list_of_configuration_ids = metrics_output.configuration_id.unique()
        metrics_df = pd.DataFrame(columns=metrics_output.columns)
        for configuration_id in list_of_configuration_ids:
            metrics_configuration = metrics_output[metrics_output.configuration_id == configuration_id]
            average_metrics_configuration = metrics_configuration.mean()
            metrics_df = pd.concat([metrics_df, average_metrics_configuration.to_frame().transpose()])

        # save as latex table
        metrics_table_filename = experiment_dir + '.tex'
        metric_table_filepath = os.path.join(self.METRIC_TABLES_OUTPUT_DIR, metrics_table_filename)
        with open(metric_table_filepath, 'w') as tf:
            tf.write(metrics_df.to_latex())

        # save as csv table
        metrics_table_filename = experiment_dir + '.csv'
        metric_table_filepath = os.path.join(self.METRIC_TABLES_OUTPUT_DIR, metrics_table_filename)
        metrics_df.to_csv(metric_table_filepath)

    def generate_confusion_matrices_for_each_configuration(self, experiment_dir, modes):
        # list all configuration folders
        experiment_result_dir_path = os.path.join(self.EXPERIMENTS_PATH, experiment_dir, self.EXPERIMENT_OUTPUT_DIR)
        configuration_dirs = os.listdir(experiment_result_dir_path)
        # for each configuration folder
        for configuration_dir in configuration_dirs:
            configuration_dir_path = os.path.join(experiment_result_dir_path, configuration_dir)
            if os.path.isdir(configuration_dir_path):
                print("Analyzing configuration {}".format(configuration_dir))
                self.generate_configuration_confusion_matrices(
                    configuration_dir, configuration_dir_path, experiment_dir, modes
                )

    def generate_configuration_confusion_matrices(
            self, configuration_dir, configuration_dir_path, experiment_dir, modes
    ):
        # intialize cumulative confusion matrix
        cumulative_confusion_matrix_train = None
        cumulative_confusion_matrix_test = None

        # list all repetitions folders
        repetition_dirs = os.listdir(configuration_dir_path)

        # for each repetition folder
        for repetition_dir in repetition_dirs:
            repetition_dir_path = os.path.join(configuration_dir_path, repetition_dir)
            if os.path.isdir(repetition_dir_path):
                confusion_matrix_train, confusion_matrix_test = self.evaluate_repetition_confusion_matrix(
                    modes, repetition_dir_path
                )
                cumulative_confusion_matrix_train = self.add_matrix_to_cumulative_matrix(
                    confusion_matrix_train, cumulative_confusion_matrix_train
                )
                cumulative_confusion_matrix_test = self.add_matrix_to_cumulative_matrix(
                    confusion_matrix_test, cumulative_confusion_matrix_test
                )

        # normalize cumulative confusion matrices
        normalized_cumulative_confusion_matrix_train = self.normalize_matrix(
            cumulative_confusion_matrix_train
        )
        normalized_cumulative_confusion_matrix_test = self.normalize_matrix(
            cumulative_confusion_matrix_test
        )

        # save normalized confusion matrix plot to svg file
        self.save_normalized_confusion_matrix_plot(
            configuration_dir, experiment_dir, modes,
            normalized_cumulative_confusion_matrix_train,
            self.CONFUSION_MATRIX_TRAIN_OUTPUT_SUFFIX
        )
        self.save_normalized_confusion_matrix_plot(
            configuration_dir, experiment_dir, modes,
            normalized_cumulative_confusion_matrix_test,
            self.CONFUSION_MATRIX_TEST_OUTPUT_SUFFIX
        )

    def evaluate_repetition_confusion_matrix(self, modes, repetition_dir_path):

        # intialize cumulative confusion matrix
        cumulative_confusion_matrix_train = None
        cumulative_confusion_matrix_test = None

        # list all folds folders
        fold_dirs = os.listdir(repetition_dir_path)

        # for each fold folder
        for fold_dir in fold_dirs:
            fold_dir_path = os.path.join(repetition_dir_path, fold_dir)

            # evaluate train confusion matrix and add to cumulative
            confusion_matrix_train = self.evaluate_fold_confusion_matrix(
                fold_dir_path, self.EXPERIMENT_TRAIN_OUTPUT_FILE, modes
            )
            cumulative_confusion_matrix_train = self.add_matrix_to_cumulative_matrix(
                confusion_matrix_train, cumulative_confusion_matrix_train
            )

            # evaluate test confusion matrix and add to cumulative
            confusion_matrix_test = self.evaluate_fold_confusion_matrix(
                fold_dir_path, self.EXPERIMENT_TEST_OUTPUT_FILE, modes
            )
            cumulative_confusion_matrix_test = self.add_matrix_to_cumulative_matrix(
                confusion_matrix_test, cumulative_confusion_matrix_test
            )

        return cumulative_confusion_matrix_train, cumulative_confusion_matrix_test

    def save_normalized_confusion_matrix_plot(
            self,
            configuration_dir,
            experiment_dir,
            modes,
            normalized_confusion_matrix,
            output_file_suffix
    ):
        plot_file_name = "confusion_matrix_{suffix}".format(suffix=output_file_suffix)
        plot_file_dir = os.path.join(self.CONFUSION_MATRIX_OUTPUT_DIR, experiment_dir, configuration_dir)
        self.check_or_create_dir(plot_file_dir)
        plot_file_path = os.path.join(plot_file_dir, plot_file_name)
        self.plot_normalized_confusion_matrix(normalized_confusion_matrix, modes)
        plt.savefig(plot_file_path)
        plt.close()

    @staticmethod
    def plot_normalized_confusion_matrix(
            normalized_confusion_matrix,
            classes,
            cmap=plt.cm.Blues
    ):

        plt.imshow(normalized_confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f'
        threshold = normalized_confusion_matrix.max() / 2.
        for i, j in itertools.product(
                range(normalized_confusion_matrix.shape[0]),
                range(normalized_confusion_matrix.shape[1])
        ):
            plt.text(j, i, format(normalized_confusion_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if normalized_confusion_matrix[i, j] > threshold else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    @staticmethod
    def add_matrix_to_cumulative_matrix(matrix, cumulative_matrix):
        if cumulative_matrix is None:
            cumulative_matrix = matrix
        else:
            cumulative_matrix = np.add(
                cumulative_matrix, matrix
            )
        return cumulative_matrix

    @staticmethod
    def evaluate_fold_confusion_matrix(fold_dir_path, ouput_file_name, classes):
        train_file_path = os.path.join(fold_dir_path, ouput_file_name)
        train_output = pd.read_csv(train_file_path)
        y_true = train_output['true']
        y_predicted = train_output['predicted']
        confusion_matrix_train = confusion_matrix(y_true, y_predicted, labels=classes)
        return confusion_matrix_train

    @staticmethod
    def normalize_matrix(matrix):
        return matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    @staticmethod
    def check_or_create_dir(dir_path):
        if os.path.isdir(dir_path) is False:
            os.makedirs(dir_path)


if __name__ == "__main__":
    analysis = TMDExperimentAnalysis()
    analysis.run()
