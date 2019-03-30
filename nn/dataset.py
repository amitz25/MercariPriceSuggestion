from __future__ import print_function
import torch
import numpy as np
import torch.utils.data as data
import pandas as pd
from nn.common import DBType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle

# https://www.kaggle.com/c/mercari-price-suggestion-challenge/data

MIN_CATEGORY_NAMES = 500
MIN_BRAND_NAMES = 100
MIN_DESC_DF = 1000


def init_dataset(config, db_type, name_vectorizer, raw_data):
    dataset = MercariDataset(config, db_type, name_vectorizer, raw_data)

    if not (db_type == DBType.Train):
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.test.batch_size, shuffle=False,
                                                  num_workers=8, drop_last=False)
    else:
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.train.batch_size, shuffle=True,
                                                  num_workers=8, drop_last=True)

    return data_loader, dataset


def limit_categorical_field(data, field_name, min_occur, raw_data):
    value_counts = raw_data[field_name].value_counts()
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

    return data.drop(field_name, axis=1), raw_data.drop(field_name, axis=1)


def preprocess_category_name(data, raw_data):
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
        raw_data[field_name] = raw_data['category_name'].apply(get_part)
        data, raw_data = limit_categorical_field(data, field_name, MIN_CATEGORY_NAMES, raw_data)

    return data.drop('category_name', axis=1)


def preprocess_brand_name(data, raw_data):
    data, raw_data = limit_categorical_field(data, 'brand_name', MIN_BRAND_NAMES, raw_data)
    return data


def train_tf_idf(kn, data, v):
    X = v.transform(data[kn])
    df = pd.DataFrame(X.toarray())
    return df


def preprocess_data(data, name_vectorizer, raw_data):
    print("Limiting category and brand names...")
    data = preprocess_category_name(data, raw_data)
    data = preprocess_brand_name(data, raw_data)

    print("Categorizing features...")
    data = pd.get_dummies(data, columns=['category_part_0_id', 'category_part_1_id',
                                         'category_part_2_id', 'brand_name_id'])

    names = train_tf_idf('name', data, name_vectorizer)

    return data, names


class MercariDataset(data.Dataset):
    def __init__(self, config, db_type, name_vectorizer, raw_data):
        self.config = config
        self.db_type = db_type
        self.longest_sent = 245
        self.vocab = np.load(config.dataset.vocab_path).item()
        self.pad_str = '<PAD>'
        self.pad_token = int(self.vocab[self.pad_str])

        self.embedding_dim = 50
        self.nb_vocab_words = len(self.vocab)

        if db_type == DBType.Test:
            db_path = config.dataset.test_path
            self.batch_size = config.test.batch_size
        elif db_type == DBType.Validation:
            db_path = config.dataset.validation_path
            self.batch_size = config.test.batch_size
        elif db_type == DBType.Train:
            db_path = config.dataset.train_path
            self.batch_size = config.train.batch_size
        else:
            raise Exception('Invalid db type')

        df = pd.read_csv(db_path, sep="\t")

        if db_type == DBType.Train:
            df = shuffle(df)

        self.df, self.names = preprocess_data(df, name_vectorizer, raw_data)
        self.dataset_size = len(df.values)
        # self.glove_model = self.load_glove_model(config.dataset.glove_path)

    def __len__(self):
        return self.dataset_size // self.batch_size * self.batch_size

    @staticmethod
    def load_glove_model(glove_file):
        print("Loading Glove Model")

        with open(glove_file, encoding="utf8") as f:
            content = f.readlines()

        model = {}
        for line in content:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding

        print("Done.", len(model), " words loaded!")

        return model

    def __getitem__(self, index):
        record = self.df.iloc[[index]]

        name = self.names.iloc[[index]].values[0]
        name = torch.tensor(name, dtype=torch.float)

        assert torch.isnan(name).max().detach().cpu().numpy() == 0

        # name = [self.vocab[word] for word in (record[1].split() if not pd.isnull(record[1]) else [self.pad_str])]
        # name_len = len(name)
        # padded_name = np.ones(self.longest_sent) * self.pad_token
        # padded_name[0:name_len] = name
        # padded_name = torch.tensor(padded_name, dtype=torch.long)
        # padded_name = self.words_embedding(padded_name)

        c_id = record.ix[:, 2].values[0]
        c_id = torch.tensor(c_id, dtype=torch.float)

        assert torch.isnan(c_id).max().detach().cpu().numpy() == 0

        price = record.ix[:, 3].values[0]
        price = torch.tensor(np.log1p(price), dtype=torch.float)

        assert torch.isnan(price).max().detach().cpu().numpy() == 0

        shipping = record.ix[:, 4].values[0]
        shipping = torch.tensor(shipping, dtype=torch.float)

        assert torch.isnan(shipping).max().detach().cpu().numpy() == 0

        tmp_desc = record.ix[:, 5].values[0]
        desc = [self.vocab[word] for word in (tmp_desc.split() if not pd.isnull(tmp_desc) else [self.pad_str])]
        desc_len = len(desc)
        padded_desc = np.ones(self.longest_sent) * self.pad_token
        padded_desc[0:desc_len] = desc
        padded_desc = torch.tensor(padded_desc, dtype=torch.long)

        assert torch.isnan(padded_desc).max().detach().cpu().numpy() == 0
        # padded_desc = self.words_embedding(padded_desc)

        # c_name = [self.vocab[word] for word in (record[3].split() if not pd.isnull(record[3]) else [self.pad_str])]
        # c_name_len = len(c_name)
        # padded_c_name = np.ones(self.longest_sent) * self.pad_token
        # padded_c_name[0:c_name_len] = c_name
        # padded_c_name = torch.tensor(padded_c_name, dtype=torch.long)

        c_name = record.ix[:, 6:353].values[0]
        c_name = torch.tensor(c_name, dtype=torch.float)

        assert torch.isnan(c_name).max().detach().cpu().numpy() == 0

        # b_name = [self.vocab[word] for word in (record[4].split() if not pd.isnull(record[4]) else [self.pad_str])]
        # b_name_len = len(b_name)
        # padded_b_name = np.ones(self.longest_sent) * self.pad_token
        # padded_b_name[0:b_name_len] = b_name
        # padded_b_name = torch.tensor(padded_b_name, dtype=torch.long)

        b_name = record.ix[:, 353:].values[0]
        b_name = torch.tensor(b_name, dtype=torch.float)

        assert torch.isnan(b_name).max().detach().cpu().numpy() == 0

        input_dict = {'name': name,
                      'cid': c_id,
                      'c_name': c_name,
                      'b_name': b_name,
                      'price': price,
                      'shipping': shipping,
                      'desc': padded_desc,
                      'desc_len': desc_len}

        return input_dict

# def word2vec(self, word):
#     if self.glove_model.__contains__(word.lower()):
#         return self.glove_model[word.lower()]
#     else:
#         return np.zeros(50)

# def __getitem__(self, index):
#     record = self.values[index]
#
#     name = [self.word2vec(x) for x in record[1].split(' ')]
#     name = torch.tensor(name, dtype=torch.float)
#
#     cid = record[2]
#     cid = torch.tensor(cid, dtype=torch.int)
#
#     c_name = [self.word2vec(x) for x in (record[3].split(' ') if not pd.isnull(record[3]) else ' ')]
#     c_name = torch.tensor(c_name, dtype=torch.float)
#
#     b_name = [self.word2vec(x) for x in (record[4].split(' ') if not pd.isnull(record[4]) else ' ')]
#     b_name = torch.tensor(b_name, dtype=torch.float)
#
#     price = record[5]
#     price = torch.tensor(price, dtype=torch.int)
#
#     shipping = record[6]
#     shipping = torch.tensor(shipping, dtype=torch.int)
#
#     desc = [self.word2vec(x) for x in (record[7].split(' ') if not pd.isnull(record[7]) else ' ')]
#     desc = torch.tensor(desc, dtype=torch.float)
#
#     input_dict = {'name': name,
#                   'cid': cid,
#                   'c_name': c_name,
#                   'b_name': b_name,
#                   'price': price,
#                   'shipping': shipping,
#                   'desc': desc}
#
#     return input_dict