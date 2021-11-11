"""
Implements the character-based decoder LSTM for predicting stems
and the tag classifier.
"""

import torch
import torch.nn as nn


def mlp_classifier_factory(input_dim, hidden_dim, num_classes, layers=1, dropout=0.0):
    """
    Implements a MLP as a single nn.Sequential module
    containing num_layers x [linear, dropout, nonlinearity]
    and eventually calculates prediction scores for each class.
    
    We do not apply a final activation function like softmax or
    log-softmax, because we use PyTorch's cross-entropy loss
    which operates on un-normalised scores.
    """
    modules = []
    # Add layers
    for layer in range(layers + 1):
        # Input dimensionality is input_dim for first layer
        # else hidden dim
        dim_in = input_dim if layer == 0 else hidden_dim
        # Output dim is hidden dim, except for the last layer
        # where we need num_classes many output values
        dim_out = num_classes if layer == layers else hidden_dim

        # Add linear transform and dropout
        modules.append(nn.Linear(dim_in, dim_out))
        modules.append(nn.Dropout(dropout))

        # Don't add nonlinearity for last ( = prediction) layer
        if layer < layers:
            modules.append(nn.ReLU())

    return nn.Sequential(*modules)
