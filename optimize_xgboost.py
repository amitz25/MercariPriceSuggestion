import argparse
import xgboost as xgb
import numpy as np
import json
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Can be used instead of just evaluate() if we want to improve stability at the cost of performance
def cross_fold(X, y, params):
    kf = KFold(n_splits=5, shuffle=True)

    all_y_test = []
    all_y_test_pred = []

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        _, Y_test_pred = evaluate(X_train, X_test, Y_train, Y_test, params)

        all_y_test.append(Y_test)
        all_y_test_pred.append(Y_test_pred)

    all_y_test = np.concatenate(all_y_test)
    all_y_test_pred = np.concatenate(all_y_test_pred)
    return np.sqrt(mean_squared_error(all_y_test, all_y_test_pred))


def evaluate(X_train, X_test, Y_train, Y_test, params):
    eval_set = [(X_test, Y_test)]

    clf = xgb.XGBRegressor(tree_method='gpu_hist', n_jobs=-1, n_estimators=10000)
    clf.set_params(**params)
    clf.fit(X_train, Y_train, eval_metric='rmse', eval_set=eval_set, early_stopping_rounds=10, verbose=False)

    Y_test_pred = clf.predict(X_test)
    loss = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
    return loss, Y_test_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--max_len', type=int, default=999999999)
    args = parser.parse_args()

    print("Loading data...")
    X, y, vectorizers = joblib.load(args.input_path)

    X = X[:args.max_len, :]
    y = y[:args.max_len]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

    # XGBoost with GPU has bugs with csr matrices
    X_train = X_train.tocsc()
    X_test = X_test.tocsc()

    def objective(param_space):
        print("Trying params:", param_space)

        loss, _ = evaluate(X_train, X_test, Y_train, Y_test, param_space)
        # loss = cross_fold(X, y, param_space)

        print("Loss:", loss)
        return {'loss': loss, 'status': STATUS_OK}

    param_space = {
        'learning_rate': hp.uniform('learning_rate', 0.1, 0.6),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1),
        'max_depth': hp.choice('max_depth', np.arange(5, 12, dtype=int)),
        'subsample': hp.uniform('subsample', 0.7, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1),
        'reg_alpha': hp.uniform('reg_alpha', 0, 0.5),
        'reg_gamma': hp.uniform('gamma', 0, 0.5),
        'lambda': hp.uniform('reg_lambda', 0.5, 1.0),
    }

    trials = Trials()
    best = fmin(fn=objective,
                space=param_space,
                algo=tpe.suggest,
                max_evals=300,
                trials=trials)

    json.dump(best, open(args.output_path, 'w'))


if __name__ == "__main__":
    main()
