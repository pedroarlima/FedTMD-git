import os
import flwr as fl
import tensorflow as tf

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Load model
#model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model = tmd
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

#Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


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

