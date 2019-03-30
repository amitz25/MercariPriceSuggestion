#!/usr/bin/env python

from __future__ import print_function
import sys
from sklearn.metrics import mean_squared_error
from nn.common import *
from nn.model import Model
from nn.dataset import *
import time
import os

MIN_NAME_DF = 1000


def train_tf_idf(min_df, kn, raw_data):
    v = TfidfVectorizer(analyzer="word", min_df=min_df)
    v = v.fit(raw_data[kn])
    return v


class Evaluator():
    def __init__(self, db_type, config, name_vectorizer, raw_df):
        super(Evaluator, self).__init__()
        self.config = config
        self.db_loader, self.dataset = init_dataset(self.config, db_type, name_vectorizer, raw_df)
        self.dataset_size = len(self.dataset)

        if torch.cuda.is_available():
            self.gpu_ids = np.array(self.config.general.gpu_ids.split(' ')).astype(np.int)
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')

        # torch.cuda.set_device(self.device)

    def eval(self, model, max_iterations=0):
        start = time.time()

        print('evaluating...')

        model.eval()

        predictions = []
        prices = []

        with torch.no_grad():
            iterations = 0

            if max_iterations == 0:
                max_iterations = self.dataset_size
            else:
                max_iterations = min(max_iterations, self.dataset_size)

            for i, data in enumerate(self.db_loader, start=0):
                if iterations % 50000 == 0:
                    print('{0} / {1}'.format(iterations, max_iterations))

                name = data['name'].to(self.device)
                cid = data['cid'].to(self.device)
                c_name = data['c_name'].to(self.device)
                b_name = data['b_name'].to(self.device)
                prices.append(data['price'].unsqueeze(1))
                shipping = data['shipping'].to(self.device)
                desc = data['desc'].to(self.device)

                desc_len = data['desc_len'].to(self.device)

                predictions.append(model.get_prediction(name, cid, c_name, b_name, shipping, desc, desc_len))

                iterations += self.config.test.batch_size

            predictions = torch.cat(predictions).cpu()
            prices = torch.cat(prices).cpu()
            err = np.sqrt(mean_squared_error(predictions.numpy(), prices.numpy()))
            print('eval error: {}'.format(err))

        model.train()

        end = time.time()

        print('evaluation took {}'.format(end - start))
        return err

    def get_prediction(self, model, max_iterations=0):
        print('evaluating...')

        model.eval()
        predictions = []

        with torch.no_grad():
            iterations = 0

            if max_iterations == 0:
                max_iterations = self.dataset_size
            else:
                max_iterations = min(max_iterations, self.dataset_size)

            for i, data in enumerate(self.db_loader, start=0):
                if iterations % 50000 == 0:
                    print('{0} / {1}'.format(iterations, max_iterations))

                name = data['name'].to(self.device)
                cid = data['cid'].to(self.device)
                c_name = data['c_name'].to(self.device)
                b_name = data['b_name'].to(self.device)
                shipping = data['shipping'].to(self.device)
                desc = data['desc'].to(self.device)

                desc_len = data['desc_len'].to(self.device)

                predictions.append(model.get_prediction(name, cid, c_name, b_name, shipping, desc, desc_len))

                iterations += self.config.test.batch_size

        return torch.cat(predictions).cpu().numpy().reshape(-1)

    def get_activations(self, model, max_iterations=0):
        print('evaluating...')

        model.eval()
        activations = []

        with torch.no_grad():
            iterations = 0

            if max_iterations == 0:
                max_iterations = self.dataset_size
            else:
                max_iterations = min(max_iterations, self.dataset_size)

            for i, data in enumerate(self.db_loader, start=0):
                if iterations % 50000 == 0:
                    print('{0} / {1}'.format(iterations, max_iterations))

                name = data['name'].to(self.device)
                cid = data['cid'].to(self.device)
                c_name = data['c_name'].to(self.device)
                b_name = data['b_name'].to(self.device)
                shipping = data['shipping'].to(self.device)
                desc = data['desc'].to(self.device)

                desc_len = data['desc_len'].to(self.device)

                activations.append(model.get_activations(name, cid, c_name, b_name, shipping, desc, desc_len))

                iterations += self.config.test.batch_size

        return torch.cat(activations).cpu().numpy()



def run_once(config, evaluator):
    model = Model(config, evaluator.dataset)
    # model = torch.nn.DataParallel(model)
    model.to(evaluator.device)
    evaluator.eval(model)


def run_seq(config, evaluator):
    for i in range(0, 21):
        print(i)
        model = Model(config, evaluator.dataset, str(i))
        # model = torch.nn.DataParallel(model)
        model.to(evaluator.device)
        evaluator.eval(model)


def main(argv):
    config = ConfigParser(os.path.join(os.path.dirname(__file__), "config.yaml"))
    raw_df = pd.read_csv(config.dataset.raw_path, sep="\t")
    name_vectorizer = train_tf_idf(MIN_NAME_DF, 'name', raw_df)
    evaluator = Evaluator(DBType.Test, config, name_vectorizer, raw_df)
    run_once(config, evaluator)
    # run_seq(config, evaluator)


if __name__ == '__main__':
    main(sys.argv)