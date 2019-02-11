import pandas as pd
import ipdb
import argparse
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack

MIN_CATEGORY_NAMES = 500
MIN_BRAND_NAMES = 100
MIN_NAME_DF = 1000
MIN_DESC_DF = 1000

xgb_params = {'gamma': 0.19710908942998961, 'reg_lambda': 0.510525715124247, 'learning_rate': 0.0773758831544977,
              'subsample': 0.7031858532224596, 'colsample_bytree': 0.8968223611898769, 'reg_alpha': 0.595353209786347,
              'max_depth': 6}


def limit_categorical_field(data, field_name, min_occur):
    value_counts = data[field_name].value_counts()
    categories = [k for k, v in value_counts.items() if v > min_occur]
    categories_dict = {k: i + 1 for i, k in enumerate(categories)}

    # Convert missing values to a separate category - 0
    categories_dict[np.nan] = 0

    # Convert all the infrequent values to a separate category
    unknown_val = len(categories_dict)

    def gen_category_id(cat):
        if cat in categories_dict:
            return categories_dict[cat]
        else:
            return unknown_val

    data[field_name + '_id'] = data[field_name].apply(gen_category_id)
    return data.drop(field_name, axis=1)


def preprocess_category_name(data):
    for i in range(4):
        def get_part(x):
            # Handle missing values (np.nan)
            if type(x) != str:
                return np.nan

            parts = x.split('/')
            if i >= len(parts):
                return np.nan
            else:
                return parts[i]

        field_name = 'category_part_' + str(i)
        data[field_name] = data['category_name'].apply(get_part)
        data = limit_categorical_field(data, field_name, MIN_CATEGORY_NAMES)

    return data.drop('category_name', axis=1)


def preprocess_brand_name(data):
    return limit_categorical_field(data, 'brand_name', MIN_BRAND_NAMES)


def handle_missing_values(data):
    # data[['category_name', 'brand_name']] = data[['category_name', 'brand_name']].fillna(value=np.nan)
    data[['name', 'item_description']] = data[['name', 'item_description']].fillna(value="")
    return data


def train_tf_idf(data, min_df):
    v = TfidfVectorizer(analyzer="word", min_df=min_df)
    X = v.fit_transform(data)
    return X, v


def preprocess_data(data):
    print("Handling missing values...")
    data = handle_missing_values(data)

    print("Limiting category and brand names...")
    data = preprocess_category_name(data)
    data = preprocess_brand_name(data)

    print("Categorizing features...")
    data = pd.get_dummies(data, columns=['category_part_0_id', 'category_part_1_id',
                                         'category_part_2_id', 'brand_name_id'])

    print("Dropping zero price rows...")
    data = data[data['price'] != 0]
    data.index = range(len(data))

    # Predict log(price) and calculate RMSE instead of RMSLE for numerical stability
    data['price'] = np.log1p(data['price'])

    return data


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str)
    parser.add_argument('--max_len', type=int, default=999999999)
    args = parser.parse_args()

    print("Loading train...")
    train = pd.read_csv('train.tsv', sep='\t')
    print("Loading test...")
    test = pd.read_csv('test.tsv', sep='\t')
    train = train[:args.max_len]

    data = preprocess_data(train)

    X = data.drop('price', axis=1)
    y = data['price']

    cross_fold(X, y)


if __name__ == "__main__":
    main()
