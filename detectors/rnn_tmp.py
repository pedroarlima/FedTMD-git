import tensorflow as tf1
import pandas as pd
import numpy as np
from detectors.nn_tmd import NeuralNetworkTMD
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell, DropoutWrapper


class RecurrentNeuralNetworkTMD(NeuralNetworkTMD):
    """
    Wrapper that uses TensorFlow to allow training and
    using a LSTM Recurrent Neural Network for
    travel mode detection through smartphone sensors
    """

    # -------------------------------- HYPERPARAMETER VALUES CONSTANTS ----------------------------------------------- #
    MINIMUM_HIDDEN_LAYERS = 1
    DEFAULT_HIDDEN_LAYERS = 1
    MAXIMUM_HIDDEN_LAYERS = None
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_TRAIN_RATIO = 0.5
    DEFAULT_VAL_RATIO = 0.5
    DEFAULT_TEST_RATIO = 0.0
    DEFAULT_BETA = 0.01
    DEFAULT_MAX_EPOCHS = 1000
    DEFAULT_OPTIMIZER = 'gradient_descent'
    DEFAULT_HIDDEN_ACTIVATION = None

    # -------------------------------- MODEL PERSISTENCE CONSTANTS --------------------------------------------------- #
    CLASSES_FILENAME = 'classes.pkl'
    LOGS_DIR = 'logs'
    DEFAULT_MODEL_PATH = 'rnn_tmd'

    # -------------------------------- TENSORFLOW VARIABLE ALIASES --------------------------------------------------- #
    LOSS_ALIAS = 'loss'
    PREDICTION_ALIAS = 'prediction'
    ACCURACY_ALIAS = 'accuracy'
    INPUT_X_ALIAS = 'input_x'

    # -------------------------------- EXTERNAL METHODS -------------------------------------------------------------- #
    def __init__(self, save_path=DEFAULT_MODEL_PATH, **kwargs):
        self.timesteps = None
        super(RecurrentNeuralNetworkTMD, self).__init__(save_path=save_path, **kwargs)

    def fit(
            self,
            data_frame: pd.DataFrame,
            timesteps=None,
            **kwargs
    ):
        if timesteps is None:
            timesteps = int(np.sqrt(len(data_frame.columns) - 1))
        self.timesteps = timesteps
        batch_size = kwargs.pop('batch_size', self.DEFAULT_BATCH_SIZE) * timesteps
        super(RecurrentNeuralNetworkTMD, self).fit(
            data_frame=data_frame,
            batch_size=batch_size,
            shuffle=kwargs.pop('shuffle', False),
            **kwargs
        )

    # -------------------------------- INTERNAL METHODS -------------------------------------------------------------- #

    @staticmethod
    def _get_hidden_layers(
            number_of_layers: int,
            number_of_neurons_by_layer: list(),
            drop_out_prob_by_layer: list(),
            activation_function_by_layer: list(),
            input_tensor: tf.Tensor,
            **kwargs
    ):
        """
        Retrieves the computational graph of the hidden layers
        :param number_of_layers:
        :param number_of_neurons_by_layer:
        :param drop_out_prob_by_layer:
        :param activation_function_by_layer:
        :param input_tensor:
        :param kwargs:
        :return:
        """

        # validate input parameters
        assert len(number_of_neurons_by_layer) == len(drop_out_prob_by_layer) == number_of_layers
        assert number_of_layers > 0

        # Define hidden layers computational graph recursively
        list_of_lstm_cells = list()
        for i in range(number_of_layers):
            list_of_lstm_cells.append(
                RecurrentNeuralNetworkTMD._get_hidden_layer(
                    input_tensor=input_tensor,
                    number_of_neurons=number_of_neurons_by_layer[i],
                    drop_out_prob=drop_out_prob_by_layer[i],
                    activation_function=activation_function_by_layer[i],
                    **kwargs
                )
            )
        multi_layer_cell = MultiRNNCell(list_of_lstm_cells)
        output, state = tf.nn.rnn(multi_layer_cell, input_tensor, dtype=tf.float32)
        output = tf.transpose(output, [1, 0, 2])
        output = tf.reshape(output, [-1, output.get_shape()[2]])
        return output

    @staticmethod
    def _get_hidden_layer(
            input_tensor: tf.Tensor,
            number_of_neurons: int,
            drop_out_prob: float,
            activation_function: str,
            **kwargs
    ):
        """
        Retrieves the computation graph of a hidden layer
        :param input_tensor:
        :param number_of_neurons:
        :param drop_out_prob:
        :param activation_function: relu, leaky_relu or maxout
        :param kwargs:
        :return:
        """

        # Define lstm cell
        hidden_layer = BasicLSTMCell(
            num_units=number_of_neurons,
            activation=activation_function,
            forget_bias=kwargs.get('forget_bias', 1.0)
        )

        # Define dropout
        if drop_out_prob > 0.0:
            return DropoutWrapper(hidden_layer, output_keep_prob=1.0 - drop_out_prob)
        else:
            return hidden_layer

    def _get_input_tensor(self, n_features, **kwargs):
        """
        Retrieves input tensor
        :param n_features:
        :param
        :return:
        """
        input_x = tf.placeholder(
            tf.float32, [None, self.timesteps, n_features], name=NeuralNetworkTMD.INPUT_X_ALIAS
        )
        return input_x

    def _get_batch_x(self, batch_index, x_train, **kwargs):
        """
        get batch_x for training
        :param batch_index:
        :param x_train:
        :param kwargs:
        :return:
        """
        batch_x = x_train[batch_index]
        batch_x = batch_x.reshape(-1, self.timesteps, batch_x.shape[1])
        return batch_x

    def _get_batch_indexes_for_accuracy_evaluation(self, batch):
        return range(0, (int(len(batch) / (self.timesteps ** 2)) * self.timesteps ** 2))

    def _validate_prediction_input_shape(self, data_frame, input_x):
        assert data_frame.shape[1] == input_x.shape[2], 'dataframe must have same shape of trained model input!'
