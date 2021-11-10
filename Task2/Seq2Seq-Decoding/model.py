"""
Helpers for building & saving model and optimizer
"""

import os
import torch
import torch.nn as nn

from torch.optim import AdamW
from encoder import Encoder
from decoders import RecurrentDecoder, mlp_classifier_factory


def build_model(config, vocabulary, tag_encoder):
    # Read hyperparameters from config
    embedding_size = config["embedding_size"]
    encoder_hidden_size = config["encoder_hidden_size"]
    encoder_max_ngram = config["encoder_max_ngram"]
    encoder_char2token_mode = config["encoder_char2token_mode"]
    decoder_hidden_dim = config["decoder_hidden_dim"]
    decoder_num_layers = config["decoder_num_layers"]
    classifier_hidden_dim = config["classifier_hidden_dim"]
    classifer_num_layers = config["classifer_num_layers"]
    dropout = config["dropout"]

    # Make encoder, stem decoder, and tag classifier
    # Encoder
    encoder = Encoder(
        len(vocabulary),
        embedding_size,
        encoder_hidden_size,
        encoder_max_ngram,
        char2token_mode=encoder_char2token_mode,
        dropout=dropout,
    )

    # Decoder (for predicting stems)
    decoder = RecurrentDecoder(
        len(vocabulary),
        embedding_size,
        encoder.output_size,
        decoder_hidden_dim,
        decoder_num_layers,
        dropout=dropout,
    )

    # Tag classifier -> morphological tags
    tag_classifier = mlp_classifier_factory(
        encoder.output_size,
        classifier_hidden_dim,
        len(tag_encoder.classes_),
        layers=classifer_num_layers,
        dropout=dropout,
    )

    model = {"encoder": encoder, "decoder": decoder, "tag_classifier": tag_classifier}
    model = nn.ModuleDict(model)

    return model


def build_optimizer(model, config):
    lr = config["learning_rate"]
    weight_decay = config.get("weight_decay", 0.01)

    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def save_model(model, optimizer, vocabulary, tag_encoder, char2index, index2char, name):
    info = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocabulary": vocabulary,
        "tag_encoder": tag_encoder,
        "char2index": char2index,
        "index2char": index2char,
    }
    if not os.path.exists("./saved_models"):
        os.makedirs("./saved_models")

    path = os.path.join(".", "saved_models", name + ".pt")
    torch.save(info, path)
