import pandas as pd
import ipdb
import argparse
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

import torch

from nn.common import *
from nn.eval import *
from nn.model import *


def activations_to_file(evaluator, output_path, config):
    model = Model(config, evaluator.dataset)
    model.to(evaluator.device)
    activations = evaluator.get_activations(model)
    joblib.dump(activations, output_path)


def main():
    config = ConfigParser(os.path.join('nn', "config.yaml"))
    raw_df = pd.read_csv(config.dataset.raw_path, sep="\t")
    name_vectorizer = train_tf_idf(MIN_NAME_DF, 'name', raw_df)

    evaluator = Evaluator(DBType.Test, config, name_vectorizer, raw_df)
    activations_to_file(evaluator, 'test_activations', config)

    evaluator = Evaluator(DBType.Train, config, name_vectorizer, raw_df)
    activations_to_file(evaluator, 'train_activations', config)


if __name__ == "__main__":
    main()
