from datasets.tmdataset import TMDataset
import pandas


def get_tmd_dataset():
    """
    Retrieves TMD dataset
    :return: tmd_dataframe: pandas.DataFrame
    """

    dataset = TMDataset()
    preprocessed_dataset = dataset.get_preprocessed_dataset()
    feature_columns = dataset.get_best_window_features()
    features_dataset = preprocessed_dataset[feature_columns]
    classes_dataset = preprocessed_dataset['target']
    full_dataset = pandas.concat([features_dataset, classes_dataset], axis=1, sort=False)
    return full_dataset
