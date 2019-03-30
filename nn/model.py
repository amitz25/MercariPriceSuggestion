import os
from nn.networks import *
import numpy as np


class Model(torch.nn.Module):
    def __init__(self, config, dataset, epoch='latest'):
        super(Model, self).__init__()

        self.config = config
        self.checkpoints_path = '{0}/{1}/'.format(self.config.general.output_path,
                                                  self.config.general.checkpoints_folder)

        if not os.path.isdir(config.general.output_path):
            os.mkdir(config.general.output_path)

        if not os.path.isdir(self.checkpoints_path):
            os.mkdir(self.checkpoints_path)

        self.classifier = Classifier(config)
        self.classifier.apply(weights_init)
        self.try_load_network(self.classifier, 'classifier', epoch)

        self.loss = self.rmse

        self.current_lr = self.config.train.lr
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.current_lr, betas=(0.5, 0.999))

        self.words_embedding = nn.Embedding(
            num_embeddings=dataset.nb_vocab_words,
            embedding_dim=dataset.embedding_dim,
            padding_idx=dataset.pad_token
        )

    def try_load_network(self, network, network_label, epoch_label):
        file_path = '{0}/{1}_{2}.dat'.format(self.checkpoints_path, epoch_label, network_label)

        if os.path.isfile(file_path):
            if torch.cuda.is_available():
                network.load_state_dict(torch.load(file_path, map_location='cuda:{0}'.format(torch.cuda.current_device())))
            else:
                network.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))
            print('{} was loaded'.format(file_path))

    def rmsle(self, h, y):
        return torch.sqrt(torch.pow(torch.log(h + 1) - torch.log(y + 1), 2).mean())

    def rmse(self, h, y):
        return torch.sqrt(torch.pow(h - y, 2).mean())

    def get_prediction(self, name, cid, c_name, b_name, shipping, desc, desc_len):
        desc = self.words_embedding(desc)
        return self.classifier(name, cid, c_name, b_name, shipping, desc, desc_len)

    def get_activations(self, name, cid, c_name, b_name, shipping, desc, desc_len):
        desc = self.words_embedding(desc)
        return self.classifier.encode_input(name, cid, c_name, b_name, shipping, desc, desc_len)

    def forward(self, name, cid, c_name, b_name, shipping, desc, desc_len, price):
        prediction = self.get_prediction(name, cid, c_name, b_name, shipping, desc, desc_len)
        loss_output = self.loss(prediction, price)

        return loss_output

    def update_learning_rate(self):
        lr = self.current_lr / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.current_lr, lr))
        self.current_lr = lr

    def save_network(self, network, network_label, epoch_label):
        if not os.path.isdir(self.checkpoints_path):
            os.mkdir(self.checkpoints_path)

        file_path = '{0}/{1}_{2}.dat'.format(self.checkpoints_path, epoch_label, network_label)
        torch.save(network.state_dict(), file_path)

    def save(self, which_epoch):
        self.save_network(self.classifier, 'classifier', which_epoch)