import pandas as pd
import argparse
import numpy as np
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack


MIN_NAME_DF = 1000
MIN_DESC_DF = 1000

xgb_params = {'gamma': 0.19710908942998961, 'reg_lambda': 0.510525715124247, 'learning_rate': 0.0773758831544977,
              'subsample': 0.7031858532224596, 'colsample_bytree': 0.8968223611898769, 'reg_alpha': 0.595353209786347,
              'max_depth': 6}


def train_tf_idf(data, min_df):
    v = TfidfVectorizer(analyzer="word", min_df=min_df)
    X = v.fit_transform(data)
    return X, v


def transform_text(data, key_name, vectorizers, min_df):
    if key_name in vectorizers:
        return vectorizers[key_name].transform(data)
    else:
        X, vectorizers[key_name] = train_tf_idf(data, min_df)
        return X


def preprocess_for_clf(X_df, Y_df, vectorizers):
    X_no_text = csr_matrix(X_df.drop(['name', 'item_description'], axis=1).values)

    X_name = transform_text(X_df['name'], 'name', vectorizers, min_df=MIN_NAME_DF)
    X_desc = transform_text(X_df['item_description'], 'desc', vectorizers, min_df=MIN_DESC_DF)

    X = hstack((X_no_text, X_name, X_desc)).tocsc()
    y = Y_df.values

    return X, y


def train_clf(X_train, Y_train, X_test, Y_test):
    print("Training CLF...")
    clf = xgb.XGBRegressor(tree_method='gpu_hist', n_jobs=-1, n_estimators=10000)
    clf.set_params(**xgb_params)

    eval_set = [(X_test, Y_test)]
    clf.fit(X_train, Y_train, eval_metric='rmse', eval_set=eval_set, early_stopping_rounds=10, verbose=False)

    Y_test_pred = clf.predict(X_test)
    return clf, np.sqrt(mean_squared_error(Y_test, Y_test_pred))


def cross_fold(X, y):
    kf = KFold(n_splits=5, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        print("Fold %d" % i)
        X_train, X_test, Y_train, Y_test = X.loc[train_index], X.loc[test_index], y.loc[train_index], y.loc[test_index]

        vectorizers = {}

        print("Engineering features for training...")
        X_train_np, Y_train_np = preprocess_for_clf(X_train, Y_train, vectorizers)

        print("Engineering features for testing...")
        X_test_np, Y_test_np = preprocess_for_clf(X_test, Y_test, vectorizers)

        print("Data shape:", X_train_np.shape)
        clf, err = train_clf(X_train_np, Y_train_np, X_test_np, Y_test_np)

        print("Error:", err)

    return clf


def train_model(X, y, activations):
    X_train, X_test, Y_train, Y_test, activations_train, activations_test = \
        train_test_split(X, y, activations, test_size=0.2)

    vectorizers = {}

    print("Engineering features for training...")
    X_train_np, Y_train_np = preprocess_for_clf(X_train, Y_train, vectorizers)

    print("Engineering features for testing...")
    X_test_np, Y_test_np = preprocess_for_clf(X_test, Y_test, vectorizers)

    activations_train = csr_matrix(activations_train).tocsc()
    activations_test = csr_matrix(activations_test).tocsc()

    X_train_np = hstack((X_train_np, activations_train))
    X_test_np = hstack((X_test_np, activations_test))

    print("Data shape:", X_train_np.shape)
    clf, err = train_clf(X_train_np, Y_train_np, X_test_np, Y_test_np)

    print("Error:", err)

    return clf, vectorizers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('activations_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--max_len', type=int, default=999999999)
    args = parser.parse_args()

    print("Loading train...")
    X, y = joblib.load(args.input_path)

    activations = joblib.load(args.activations_path)

    data_size = activations.shape[0]

    X = X[:data_size]
    y = y[:data_size]

    model, vectorizers = train_model(X, y, activations)

    joblib.dump((model, vectorizers), args.output_path)


if __name__ == "__main__":
    main()
