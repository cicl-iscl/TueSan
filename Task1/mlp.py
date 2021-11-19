"""
Implements multiple standard neural building blocks that are used in
various more sophisticated models:

  * MLP (aka multi-layer perceptron)
    = sequence of Dropout, Linear, elementwise nonlinearity
  * MLC (aka multi-layer convolution)
    = sequence of Dropout, 1d-convolution, batch norm, nonlinearity
  * Residual MLP
    = like MLP, but add input to output (enforce same dimension)
  * BiLSTM
    = mainly wraps packing/unpacking of sequences
"""


import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import numpy as np


class ConvLayerNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layer_norm = nn.LayerNorm((num_features,))

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.layer_norm(x)
        x = x.transpose(-1, -2)
        return x


def make_linear_transform(in_size, out_size):
    """
    Factory for linear transforms:
      * initialises weights using Xavier normal initialisation
      * sets biases to 0
    """
    linear = nn.Linear(in_size, out_size)
    nn.init.xavier_normal_(linear.weight)
    nn.init.zeros_(linear.bias)

    return linear


def mlp(input_dim, hidden_dim, num_layers, output_dim=None, dropout=0.0):
    """
    Factory function for creating MLPs.
    Returns a nn.Sequential module containing `num_layers` nonlinear transforms
    and 1 final linear transform that projects to the given output dimension.
    """

    # If output dim not given, assume hidden dim as output dim
    output_dim = output_dim if output_dim is not None else hidden_dim

    modules = []
    for layer in range(num_layers + 1):
        in_size = input_dim if layer == 0 else hidden_dim
        out_size = output_dim if layer == num_layers else hidden_dim

        modules.append(nn.Dropout(p=dropout))
        modules.append(make_linear_transform(in_size, out_size))

        # Nonlinearity only for hidden layers
        if layer < num_layers:
            modules.append(nn.ReLU())
            # modules.append(nn.LayerNorm((out_size,)))

    return nn.Sequential(*modules)


def mlc(input_dim, hidden_dim, filter_size, num_layers, output_dim=None, dropout=0.0):
    """
    Factory function for creating multiple convolution layers.
    """
    output_dim = output_dim if output_dim is not None else hidden_dim

    modules = []
    for layer in range(num_layers):
        in_size = input_dim if layer == 0 else hidden_dim
        out_size = output_dim if layer + 1 == num_layers else hidden_dim

        modules.append(nn.Dropout(p=dropout))
        modules.append(nn.Conv1d(in_size, out_size, filter_size, padding="same"))
        modules.append(nn.BatchNorm1d(out_size)),
        modules.append(nn.ReLU())

        # if layer + 1 < num_layers:
        #    modules.append(ConvLayerNorm(out_size))

    return nn.Sequential(*modules)


class ResidualMLP(nn.Module):
    """
    Residual MLP:
    Adds input of MLP to output (skip-connection)
    """

    def __init__(self, input_dim, num_layers, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.mlp = mlp(
            input_dim, hidden_dim, num_layers, output_dim=input_dim, dropout=dropout
        )

    def forward(self, x):
        return self.mlp(x) + x


class LSTM(nn.Module):
    """
    Wrapper around LSTM that handles packing/unpacking of padded sequences
    """

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()

        # Instantiate LSTM
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

    def forward(self, inputs, lengths):
        # Pack sentences
        lengths = torch.clamp(lengths, min=1).cpu()
        inputs = pack_padded_sequence(
            inputs, lengths, batch_first=True, enforce_sorted=False
        )
        # Apply LSTM
        inputs, _ = self.lstm(inputs)
        # Unpack sentences
        inputs, _ = pad_packed_sequence(inputs, batch_first=True)

        return inputs
