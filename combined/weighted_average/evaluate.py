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

from train import preprocess_for_clf


def get_xgb_predictions(X, y, model_path):
    model, vectorizers = joblib.load(model_path)

    print("Engineering features for XGBoost...")
    X_np, y_np = preprocess_for_clf(X, y, vectorizers)
    return model.predict(X_np)


def get_nn_predictions():
    config = ConfigParser(os.path.join('nn', "config.yaml"))
    raw_df = pd.read_csv(config.dataset.raw_path, sep="\t")
    name_vectorizer = train_tf_idf(MIN_NAME_DF, 'name', raw_df)
    evaluator = Evaluator(DBType.Test, config, name_vectorizer, raw_df)

    model = Model(config, evaluator.dataset)
    model.to(evaluator.device)
    return evaluator.get_prediction(model)


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('xgb_path', type=str)
    parser.add_argument('test_path', type=str)
    args = parser.parse_args()

    print("Loading data...")
    X, y = joblib.load(args.test_path)

    start = time.time()

    nn_predictions = get_nn_predictions()
    xgb_predictions = get_xgb_predictions(X, y, args.xgb_path)

    end = time.time()
    print("Evaluation took", end-start)

    # nn prediction's size is cut by ~8 since it has to divide by batch size
    nn_pred_size = nn_predictions.shape[0]
    xgb_predictions = xgb_predictions[:nn_pred_size]
    combined_predictions = 0.8 * xgb_predictions + 0.2 * nn_pred_size
    y = y[:nn_pred_size]

    print("NN err:", rmsle(y, nn_predictions))
    print("XGB err:", rmsle(y, xgb_predictions))
    print("Combined err:", rmsle(y, combined_predictions))


if __name__ == "__main__":
    main()
