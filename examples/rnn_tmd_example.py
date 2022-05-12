from detectors.rnn_tmd import RecurrentNeuralNetworkTMD
from examples.util import get_tmd_dataset
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    # load dataset
    df = get_tmd_dataset()
    travel_mode_column = 'target'

    # train and save model
    rnn_tmd = RecurrentNeuralNetworkTMD()
    rnn_tmd.fit(
         data_frame=df.copy(),
         travel_mode_col=travel_mode_column,
         shuffle=False,
         n_hidden_layers=3,
         beta=0.01,
         optimizer='adagrad',
         batch_size=128,
         max_epochs=200,
         learning_rate=0.01,
         timesteps=1,
         exponential_decay=True
    )

    # evaluate accuracy
    labeled_modes = df.pop(travel_mode_column)
    detected_modes = rnn_tmd.predict(df.copy())
    print('Full Data Accuracy', accuracy_score(labeled_modes, detected_modes))
