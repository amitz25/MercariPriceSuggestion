import pandas as pd
import ipdb
import argparse
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

from train import preprocess_for_clf


def evaluate_model(model, vectorizers, X, y):
    print("Engineering features...")
    X_np, y_np = preprocess_for_clf(X, y, vectorizers)

    y_pred = model.predict(X_np)
    err = np.sqrt(mean_squared_error(y_np, y_pred))
    print("Error:", err)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('test_path', type=str)
    args = parser.parse_args()

    model, vectorizers = joblib.load(args.model_path)

    print("Loading data...")
    X, y = joblib.load(args.test_path)

    evaluate_model(model, vectorizers, X, y)


if __name__ == "__main__":
    main()
