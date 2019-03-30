import pandas as pd
import numpy as np
from sklearn.externals import joblib

MIN_CATEGORY_NAMES = 500
MIN_BRAND_NAMES = 100


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


def main():
    print("Loading train...")
    train = pd.read_csv('data/train.tsv', sep='\t')

    print("Loading test...")
    test = pd.read_csv('data/test.tsv', sep='\t')

    data = preprocess_data(pd.concat([train, test]))

    train = data[:len(train)]
    test = data[len(train):]

    X_train = train.drop('price', axis=1)
    y_train = train['price']

    X_test = test.drop('price', axis=1)
    y_test = test['price']

    print("Dumping data...")
    joblib.dump((X_train, y_train), 'data/train_preprocessed.joblib')
    joblib.dump((X_test, y_test), 'data/test_preprocessed.joblib')


if __name__ == "__main__":
    main()
