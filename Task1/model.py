import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from torch.optim import AdamW
import torch.nn


def build_model(config, vocabulary):
    # Read hyperparameters from config
    embedding_dim = config["embedding_dim"]
    hidden_dim = config["hidden_dim"]

    return SegmenterModel(len(vocabulary), embedding_dim, hidden_dim)


class SegmenterModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(2 * hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _sequence_mask(self, lengths):
        # from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
        batch_size = lengths.shape[0]
        max_len = torch.max(lengths).item()
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
        return seq_range_expand < seq_length_expand

    def forward(self, input):
        batch_size = input.shape[0]
        lengths = torch.sum(input != 0, dim=-1).long().cpu()
        mask = self._sequence_mask(lengths).to(input.device)

        input = self.embedding(input)
        input = pack_padded_sequence(
            input, lengths, batch_first=True, enforce_sorted=False
        )
        input, _ = self.rnn(input)
        input, lengths = pad_packed_sequence(input, batch_first=True)
        predictions = self.projection(input)
        predictions = predictions.reshape(batch_size, -1)
        masked_predictions = predictions * mask

        return masked_predictions


def build_optimizer(model, config):

    return AdamW(model.parameters())


def get_loss(config):
    return getattr(torch.nn, config["loss"])()


def save_model(model, optimizer, vocabulary, char2index, index2char, name):
    info = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocabulary": vocabulary,
        "char2index": char2index,
        "index2char": index2char,
    }
    if not os.path.exists("./saved_models"):
        os.makedirs("./saved_models")

    path = os.path.join(".", "saved_models", name + ".pt")
    torch.save(info, path)
