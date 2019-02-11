import argparse
import pandas as pd
from train import preprocess_data, preprocess_for_clf
from sklearn.externals import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()

    print("Loading data...")
    data = pd.read_csv(args.input_path, sep='\t')
    data = preprocess_data(data)

    X_df = data.drop('price', axis=1)
    y_df = data['price']

    vectorizers = {}

    print("Training TFIDF...")
    X, y = preprocess_for_clf(X_df, y_df, vectorizers)
    print("Data:", X.shape)
    joblib.dump((X, y, vectorizers), args.output_path)


if __name__ == "__main__":
    main()