import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.init as init


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()

        self.config = config
        self.rnn_input_size = 50
        self.dropout_prob = 0.5

        if self.config.rnn_encoder.bi_directional:
            raise Exception('b-directional rnn is not supported yet.')
            self.num_dir = 2
        else:
            self.num_dir = 1

        if self.config.rnn_encoder.rnn_type == 'gru':
            self.rnn_encoder = nn.GRU(input_size=self.rnn_input_size,
                                      hidden_size=self.config.rnn_encoder.hidden_size,
                                      num_layers=self.config.rnn_encoder.num_layers,
                                      bidirectional=self.config.rnn_encoder.bi_directional,
                                      dropout=self.dropout_prob)
        elif self.config.rnn_encoder.rnn_type == 'lstm':
            self.rnn_encoder = nn.LSTM(input_size=self.rnn_input_size,
                                       hidden_size=self.config.rnn_encoder.hidden_size,
                                       num_layers=self.config.rnn_encoder.num_layers,
                                       bidirectional=self.config.rnn_encoder.bi_directional,
                                       dropout=self.dropout_prob)
        else:
            raise Exception("Invalid rnn model")

        self.attn_out_dim = 128
        self.desc_attention = Attention(self.config.rnn_encoder.hidden_size, self.attn_out_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(450, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.name_encoder = nn.Sequential(
            nn.Linear(973, 512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(512, 128),
            nn.ReLU(),
        )

        self.c_name_encoder = nn.Sequential(
            nn.Linear(347, 128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.b_name_encoder = nn.Sequential(
            nn.Linear(597, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def encode_input(self, name, cid, c_name, b_name, shipping, desc, desc_len):
        encoded_desc = self.encode_sequence(desc, desc_len, self.desc_attention)
        encoded_name = self.name_encoder(name)
        encoded_c_name = self.c_name_encoder(c_name)
        encoded_b_name = self.b_name_encoder(b_name)
        return torch.cat(
            (encoded_name, encoded_b_name, encoded_c_name, encoded_desc, shipping.unsqueeze(1), cid.unsqueeze(1)),
            dim=1)

    def forward(self, name, cid, c_name, b_name, shipping, desc, desc_len):
        input_to_classifier = self.encode_input(name, cid, c_name, b_name, shipping, desc, desc_len)
        result = self.classifier(input_to_classifier)

        return result

    def encode_sequence(self, sequence, len, attention):
        sequence = sequence.permute(1, 0, 2)
        len, perm_idx = len.sort(0, descending=True)
        sequence = sequence[:, perm_idx]
        packed = torch.nn.utils.rnn.pack_padded_sequence(sequence, len)
        encoded, hidden = self.rnn_encoder(packed)
        encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(encoded)
        _, unperm_idx = perm_idx.sort(0)
        encoded = encoded[:, unperm_idx]
        hidden = hidden[-1, unperm_idx]
        result = attention(hidden, encoded)

        return result


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, out_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.out_dim = out_dim

        self.attn = nn.Linear(out_dim + enc_hid_dim, out_dim)
        self.v = nn.Parameter(torch.rand(out_dim))

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        a = F.softmax(attention, dim=1)
        a = a.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)

        return weighted[0]