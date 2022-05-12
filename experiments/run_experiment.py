from experiments.tmd_experiment_base import TMDExperiment
from os import path
from detectors.rnn_tmd import RecurrentNeuralNetworkTMD

experiment = TMDExperiment(
    experiment_path=path.abspath(path.dirname(__file__)),
    detector_type=RecurrentNeuralNetworkTMD
)
experiment.run()
