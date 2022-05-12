import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.externals import joblib
from detectors.tmd_base import TravelModeDetector


class NeuralNetworkTMD(TravelModeDetector):
    """
    Wrapper that uses TensorFlow to allow training and
    using a feedforward neural network for
    travel mode detection through smartphone sensors
    """

    # -------------------------------- HYPERPARAMETER VALUES CONSTANTS ----------------------------------------------- #
    MINIMUM_HIDDEN_LAYERS = 0
    DEFAULT_HIDDEN_LAYERS = 1
    MAXIMUM_HIDDEN_LAYERS = 1
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_TRAIN_RATIO = 0.5
    DEFAULT_VAL_RATIO = 0.5
    DEFAULT_TEST_RATIO = 0.0
    DEFAULT_BETA = 0.01
    DEFAULT_MAX_EPOCHS = 1000
    DEFAULT_OPTIMIZER = 'adam'
    DEFAULT_HIDDEN_ACTIVATION = 'relu'

    # -------------------------------- MODEL PERSISTENCE CONSTANTS --------------------------------------------------- #
    CLASSES_FILENAME = 'classes.pkl'
    LOGS_DIR = 'logs'
    DEFAULT_MODEL_PATH = 'nn_tmd'

    # -------------------------------- TENSORFLOW VARIABLE ALIASES --------------------------------------------------- #
    LOSS_ALIAS = 'loss'
    PREDICTION_ALIAS = 'prediction'
    ACCURACY_ALIAS = 'accuracy'
    INPUT_X_ALIAS = 'input_x'

    # -------------------------------- EXTERNAL METHODS -------------------------------------------------------------- #
    def __init__(self, save_path=DEFAULT_MODEL_PATH, **kwargs):
        super(NeuralNetworkTMD, self).__init__(save_path=save_path, **kwargs)

    def fit(
            self,
            data_frame: pd.DataFrame,
            travel_mode_column: str = 'target',
            shuffle: bool = True,
            train_ratio: float = DEFAULT_TRAIN_RATIO,
            val_ratio: float = DEFAULT_VAL_RATIO,
            test_ratio: float = DEFAULT_TEST_RATIO,
            one_hot_encode: bool = False,
            fill_nan_with_mean: bool = True,
            convert_modes_to_numbers: bool = True,
            n_hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
            number_of_neurons_by_layer: list() = None,
            drop_out_prob_by_layer: list() = None,
            activation_function_by_layer: list() = None,
            beta: float = DEFAULT_BETA,
            optimizer: str = DEFAULT_OPTIMIZER,
            batch_size: int = DEFAULT_BATCH_SIZE,
            replace: bool = False,
            max_epochs: int = DEFAULT_MAX_EPOCHS,
            display_step: int = 20,
            hidden_units_decay: int = 2,
            **kwargs
    ):
        """
        Pre process input and train an Artificial Neural Network for travel mode detection
        using mini-batch Backpropagation with l2 regularization and dropout
        :param data_frame: data frame containing input features and labels
        :param travel_mode_column: name of the data data_frame column containing the labels
        :param shuffle: If True, the data_frame entries will be shuffled
        :param train_ratio: fraction of samples of the dataframe that are using for training
        :param val_ratio: fraction of samples of the dataframe that are using for validation
        :param test_ratio: fraction of samples of the dataframe that are using for testing
        :param one_hot_encode: If True, labels will be one hot encoded
        :param fill_nan_with_mean: If True, features with NaN value will be replaced by the mean
        :param convert_modes_to_numbers: If True, travel mode labels will be converted to integers
        :param n_hidden_layers: Number of hidden layers in the neural network
        :param number_of_neurons_by_layer: Number of neurons in each layer
        :param activation_function_by_layer: Activation function on each layer somation ('relu', 'leaky_relu', 'maxout')
        :param drop_out_prob_by_layer: Probability of neuron drop out in each layer
        :param beta: regularization constant for L2 regularization in loss function
        :param optimizer: Optimization Algorithm used to train model ('adam', 'adagrad', 'momentum', 'gradient_descent')
        :param batch_size: size of the batch used in mini-batch training of the model
        :param replace: If True, batchs will be constructed with replacement of samples
        :param max_epochs: Maximum number of iterations during training
        :param display_step: After how many epochs the loss and accuracy are displayed
        :param hidden_units_decay: Division factor applied to the default number of hidden units
        :param kwargs: additional arguments
        :return:
        """

        assert n_hidden_layers >= self.MINIMUM_HIDDEN_LAYERS
        if self.MAXIMUM_HIDDEN_LAYERS is not None:
            assert n_hidden_layers <= self.MAXIMUM_HIDDEN_LAYERS

        # partition data frame into train, validation and test
        x_test, x_train, x_val, y_test, y_train, y_val = self.get_preprocessed_partitions(
            data_frame=data_frame,
            travel_mode_column=travel_mode_column,
            shuffle=shuffle,
            one_hot_encode=one_hot_encode,
            fill_nan_with_mean=fill_nan_with_mean,
            convert_modes_to_numbers=convert_modes_to_numbers,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

        # get train dataset dimensions
        n_samples = x_train.shape[0]
        n_features = x_train.shape[1]

        # set and check neurons by layer
        number_of_neurons_by_layer = self._parse_list_of_values_by_layer(
            list_of_values=number_of_neurons_by_layer,
            default_value=n_features,
            number_of_layers=n_hidden_layers,
            division_factor=hidden_units_decay
        )

        # set and check drop out probability by layer
        drop_out_prob_by_layer = self._parse_list_of_values_by_layer(
            list_of_values=drop_out_prob_by_layer,
            default_value=0.0,
            number_of_layers=n_hidden_layers
        )

        # set and check activation function by layer
        activation_function_by_layer = self._parse_list_of_values_by_layer(
            list_of_values=activation_function_by_layer,
            default_value=self.DEFAULT_HIDDEN_ACTIVATION,
            number_of_layers=n_hidden_layers
        )

        # get batch size and output size
        batch_size = min(batch_size, n_samples)
        n_output_labels = len(list(data_frame[travel_mode_column].unique()))

        tf.reset_default_graph()

        # Define Input Layer
        input_x = self._get_input_tensor(n_features)
        input_y = self._get_output_tensor()

        # Define Hidden Layers
        if n_hidden_layers > 0:
            hidden_layers = self._get_hidden_layers(
                input_tensor=input_x,
                number_of_layers=n_hidden_layers,
                number_of_neurons_by_layer=number_of_neurons_by_layer,
                drop_out_prob_by_layer=drop_out_prob_by_layer,
                activation_function_by_layer=activation_function_by_layer,
                **kwargs
            )
            output_layer_input = hidden_layers
            output_layer_input_width = number_of_neurons_by_layer[-1]
        else:
            output_layer_input = input_x
            output_layer_input_width = n_features

        # Define Output layer
        output_logits, output_weights = self._get_output_layer(
            n_output_labels, output_layer_input, output_layer_input_width
        )

        # Define Loss Function
        loss_op = self._get_loss(beta, input_y, output_logits, output_weights)

        tf.summary.histogram(self.LOSS_ALIAS, loss_op)

        # Define Optimizer
        train_op = self._get_training_operation(loss_op, optimizer, max_epochs=max_epochs, **kwargs)

        # Define Predict Function
        prediction_op = self._get_output_activation(output_logits)

        tf.summary.histogram(self.PREDICTION_ALIAS, prediction_op)

        # Define Evaluation Function
        mean_accuracy_op = self._get_accuracy(input_y, prediction_op)
        tf.summary.histogram(self.ACCURACY_ALIAS, mean_accuracy_op)

        # Initializing tf session
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as session:

            session.run(init)
            logs_path = os.path.join(self.save_path, self.LOGS_DIR)
            train_writer = tf.summary.FileWriter(logs_path, session.graph)

            # Training cycle
            for epoch in range(max_epochs):
                epoch_loss = 0.0
                epoch_accuracy = 0.0
                n_batches = int(n_samples / batch_size)
                merge = tf.summary.merge_all()

                # Loop over all batches
                for i in range(n_batches):
                    batch_index = np.random.choice(n_samples, batch_size, replace=replace)
                    batch_x = self._get_batch_x(batch_index, x_train)
                    batch_y = self._get_batch_y(batch_index, y_train)

                    # Run optimization op (backprop) and cost op (to get loss value)
                    summary, _, loss, epoch_accuracy = session.run(
                        fetches=[merge, train_op, loss_op, mean_accuracy_op],
                        feed_dict={input_x: batch_x, input_y: batch_y}
                    )

                    # Compute average loss and save summary
                    epoch_loss += loss / n_batches
                    train_writer.add_summary(summary, i)


                # Display logs per epoch step
                if epoch % display_step == 0:
                    print(
                        "Epoch:", '%03d' % epoch,
                        "loss={:.6f}".format(epoch_loss),
                        "accuracy={:.4f}".format(epoch_accuracy)
                    )

            # evaluate validation accuracy
            if val_ratio > 0.0:
                epoch_accuracy = self._evaluate_accuracy(
                    session=session,
                    input_x=input_x,
                    input_y=input_y,
                    accuracy_op=mean_accuracy_op,
                    prediction_op=prediction_op,
                    x=x_val,
                    y=y_val
                )
                print("Validation accuracy:", epoch_accuracy)

            # evaluate test accuracy
            if test_ratio > 0.0:
                epoch_accuracy = self._evaluate_accuracy(
                    input_x, input_y, mean_accuracy_op, prediction_op, session, x_test, y_test
                )
                print("Test accuracy:", epoch_accuracy)

            # save model and classes mapping to numbers
            saver.save(session, os.path.join(self.save_path, 'model'))
            joblib.dump(self.classes, os.path.join(self.save_path, self.CLASSES_FILENAME))

            self.model = prediction_op
            self.model_byte_size = session.graph_def.ByteSize()

    def predict(self, data_frame: pd.DataFrame, batch_size=DEFAULT_BATCH_SIZE, verbose=0):
        """
        Detect travel mode of samples in a dataframe
        :param data_frame:
        :param batch_size:
        :param verbose:
        :return:
        """

        # restore checkpoint
        checkpoint = tf.train.latest_checkpoint(self.save_path)
        tf.reset_default_graph()

        # start session
        with tf.Session() as session:

            session.run(tf.global_variables_initializer())

            # restore graph
            graph = tf.get_default_graph()
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint))

            # restore variables
            saver.restore(session, checkpoint)
            prediction = graph.get_tensor_by_name('{}:0'.format(self.PREDICTION_ALIAS))
            input_x = graph.get_tensor_by_name('{}:0'.format(self.INPUT_X_ALIAS))

            # restore classes
            self.classes = joblib.load(os.path.join(self.save_path, self.CLASSES_FILENAME))
            self.classes2string = {}
            self.classes2number = {}
            for i in range(len(self.classes)):
                c = self.classes[i]
                self.classes2string[i] = c
                self.classes2number[c] = i

            # preprocess input data
            self._validate_prediction_input_shape(data_frame, input_x)
            data_frame = self.fill_nan_with_mean(data_frame)

            # detect modes in batches
            samples = data_frame.values
            n_samples = samples.shape[0]
            n_batches = int(n_samples / batch_size)
            predictions = list()
            for batch_index in range(n_batches + 1):
                batch_begin = batch_index * batch_size
                batch_end = min(batch_begin + batch_size, n_samples)
                batch_index = range(batch_begin, batch_end)
                batch_x = self._get_batch_x(batch_index, samples)

                predicted = session.run(
                    fetches=[prediction],
                    feed_dict={input_x: batch_x}
                )
                predictions.extend(predicted[0].tolist())

            # convert modes numbers to labels
            predictions = pd.DataFrame(predictions)
            predictions = self.convert_numbers_to_classes(predictions)

            return predictions

# ------------------------------------- INTERNAL METHODS ------------------------------------------------------------- #

    @staticmethod
    def _validate_prediction_input_shape(data_frame, input_x):
        assert data_frame.shape[1] == input_x.shape[1], \
            'dataframe must have same shape of trained model input!'

    @staticmethod
    def _parse_list_of_values_by_layer(default_value, number_of_layers, list_of_values, division_factor=None):
        """
        If list is not passed, populate with default value.
        Otherwise, check if lenght is equal to number of layers
        :param default_value:
        :param number_of_layers:
        :param list_of_values:
        :param division_factor: Division factor applied on default value by layer
        :return:
        """
        if list_of_values is None:
            list_of_values = list()
            for i in range(number_of_layers):
                list_of_values.append(default_value)
                if division_factor is not None:
                    default_value = int(default_value/division_factor)
        else:
            assert len(list_of_values) == number_of_layers
        return list_of_values

    @staticmethod
    def _get_accuracy(input_y, prediction):
        """
        Retrieves computation graph of accuracy function
        :param input_y:
        :param prediction:
        :return:
        """
        accuracy = tf.cast(tf.equal(prediction, input_y), tf.float32)  # Needs to be float to round the mean
        mean_accuracy = tf.reduce_mean(accuracy, name='accuracy')
        return mean_accuracy

    @staticmethod
    def _get_output_activation(output_logits):
        """
        Retrieves computation graph of output activation function
        :param output_logits:
        :return:
        """
        prediction = tf.cast(tf.argmax(output_logits, 1), tf.int32, name='prediction')
        return prediction

    @staticmethod
    def _get_output_tensor():
        """
        Retrieves output tensor
        :return:
        """
        input_y = tf.placeholder(tf.int32, [None], name='input_y')
        return input_y

    @staticmethod
    def _get_input_tensor(n_features, **kwargs):
        """
        Retrieves input tensor
        :param n_features:
        :param kwargs:
        :return:
        """
        input_x = tf.placeholder(tf.float32, [None, n_features], name=NeuralNetworkTMD.INPUT_X_ALIAS)
        return input_x

    @staticmethod
    def _get_loss(
            beta,
            input_y,
            logits,
            output_weights,
            loss_function='sparse_softmax_cross_entropy_with_logits',
            regularization='l2_loss'
    ):
        """
        Retrieves computational graph of loss function
        :param beta:
        :param input_y:
        :param logits:
        :param output_weights:
        :param loss_function:
        :param regularization:
        :return:
        """

        # Define loss function
        if loss_function == 'sparse_softmax_cross_entropy_with_logits':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=input_y
            )
        elif loss_function == 'softmax_cross_entropy_with_logits':
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=input_y
            )
        else:
            raise NotImplementedError

        # Define regularization
        if regularization == 'l2_loss':
            regularizer = tf.nn.l2_loss(output_weights)
        else:
            regularizer = 0

        return tf.reduce_mean(loss + beta * regularizer, name='loss')

    @staticmethod
    def _get_training_operation(loss: tf.Tensor, optimizer: str, **kwargs):
        """
        Retrieves the computational graph of the training operation
        :param loss:
        :param optimizer: adam, adagrad, momentum or gradient_descent
        :param kwargs:
        :return:
        """
        initial_learning_rate = kwargs.get('learning_rate', 0.01)

        if kwargs.get('exponential_decay', False):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                learning_rate=initial_learning_rate,
                global_step=global_step,
                decay_steps=kwargs.get('max_epochs', None),
                decay_rate=kwargs.get('decay_rate', 0.96),
                staircase=kwargs.get('staircase', True)
            )
        else:
            learning_rate = initial_learning_rate

        if optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=kwargs.get('beta1', 0.9),
                beta2=kwargs.get('beta2', 0.999),
                epsilon=kwargs.get('epsilon', 1e-08)
            )
        elif optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                learning_rate=learning_rate,
                initial_accumulator_value=kwargs.get('initial_accumulator_value', 0.1),
                use_locking=kwargs.get('use_locking', False)
            )
        elif optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=kwargs.get('momentum', 0.01),
                use_locking=kwargs.get('use_locking', False),
                use_nesterov=kwargs.get('use_nesterov', False)
            )
        elif optimizer == 'gradient_descent':
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate,
                use_locking=kwargs.get('use_locking', False)
            )
        else:
            raise NotImplementedError

        return optimizer.minimize(loss)

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
        hidden_layer = None
        for i in range(number_of_layers):
            hidden_layer = NeuralNetworkTMD._get_hidden_layer(
                input_tensor=input_tensor,
                number_of_neurons=number_of_neurons_by_layer[i],
                drop_out_prob=drop_out_prob_by_layer[i],
                activation_function=activation_function_by_layer[i],
                **kwargs
            )
            input_tensor = hidden_layer
        return hidden_layer

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

        # Define somation
        weigths = tf.Variable(tf.random_normal([int(input_tensor.shape[1]), number_of_neurons]))
        biases = tf.Variable(tf.zeros([number_of_neurons]))
        somation = tf.nn.bias_add(tf.matmul(input_tensor, weigths), biases)

        hidden_layer = NeuralNetworkTMD._get_activation(
            activation_function=activation_function,
            somation=somation,
            **kwargs
        )

        # Define dropout
        if drop_out_prob > 0.0:
            return tf.nn.dropout(hidden_layer, drop_out_prob)
        else:
            return hidden_layer

    def _get_output_layer(self, n_output_labels, output_layer_input, output_layer_input_width):
        """
        Retrieves output layer computations
        :param n_output_labels:
        :param output_layer_input:
        :param output_layer_input_width:
        :return:
        """
        output_weights = self._get_output_weights(n_output_labels, output_layer_input_width)
        output_biases = self._get_output_biases(n_output_labels)
        output_logits = tf.nn.bias_add(tf.matmul(output_layer_input, output_weights), output_biases)
        return output_logits, output_weights

    @staticmethod
    def _get_output_biases(n_output_labels):
        """
        Retrieves weights of output layer
        :param n_output_labels:
        :return:
        """
        return tf.Variable(tf.zeros([n_output_labels]))

    @staticmethod
    def _get_output_weights(n_output_labels, output_layer_input_width):
        """
        Retrieves biases of output layer
        :param n_output_labels:
        :param output_layer_input_width:
        :return:
        """
        return tf.Variable(tf.random_normal([output_layer_input_width, n_output_labels]))

    @staticmethod
    def _get_activation(
            somation: tf.Tensor,
            activation_function: str,
            **kwargs
    ):
        """
        Retrieves the activation function computational graph
        :param somation:
        :param activation_function:
        :param kwargs:
        :return:
        """
        if activation_function == 'relu':
            activation = tf.nn.relu(somation)
        elif activation_function == 'leaky_relu':
            activation = tf.nn.leaky_relu(
                somation,
                alpha=kwargs.get('leaky_relu_alpha', 0.2)
            )
        elif activation_function == 'maxout':
            maxout_num_units = kwargs.get('maxout_num_units', int(somation.shape[1]))
            activation = tf.contrib.layers.maxout(
                somation,
                num_units=maxout_num_units,
                axis=kwargs.get('maxout_axis', -1),
                scope=kwargs.get('maxout_scope', None)
            )
            activation.set_shape((None, maxout_num_units))
        else:
            raise NotImplementedError
        return activation

    def _evaluate_accuracy(
            self,
            session: tf.Session,
            input_x: tf.Tensor,
            input_y: tf.Tensor,
            accuracy_op: tf.Tensor,
            prediction_op: tf.Tensor,
            x: np.ndarray,
            y: np.ndarray,
    ):
        """
        Evaluates accuracy metric
        :param session:
        :param input_x:
        :param input_y:
        :param accuracy_op:
        :param prediction_op:
        :param x:
        :param y:
        :return:
        """
        batch_x = self._get_batch_x(self._get_batch_indexes_for_accuracy_evaluation(x), x)
        batch_y = self._get_batch_y(self._get_batch_indexes_for_accuracy_evaluation(y), y)
        predicted, epoch_accuracy = session.run(
            fetches=[prediction_op, accuracy_op],
            feed_dict={input_x: batch_x, input_y: batch_y}
        )
        return epoch_accuracy

    @staticmethod
    def _get_batch_indexes_for_accuracy_evaluation(batch):
        return range(0, len(batch))

    @staticmethod
    def _get_batch_x(batch_index, x_train, **kwargs):
        """
        get batch_x for training
        :param batch_index:
        :param x_train:
        :param kwargs:
        :return:
        """
        batch_x = x_train[batch_index]
        return batch_x

    @staticmethod
    def _get_batch_y(batch_index, y_train):
        """
        get batch_y for training
        :param batch_index:
        :param y_train:
        :return:
        """
        return y_train[batch_index].flatten()
