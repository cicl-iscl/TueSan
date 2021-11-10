"""
Helpers for building & saving model and optimizer
"""

import os
import torch
import torch.nn as nn

from torch.optim import AdamW
from encoder import Encoder
from decoders import mlp_classifier_factory


def build_model(config, vocabulary, classes, names):
    # Read hyperparameters from config
    embedding_size = config["embedding_size"]
    encoder_hidden_size = config["encoder_hidden_size"]
    encoder_max_ngram = config["encoder_max_ngram"]
    encoder_char2token_mode = config["encoder_char2token_mode"]
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

    # Rule/Tag classifier -> morphological rules/tags
    model = {"encoder": encoder}
    model = nn.ModuleDict(model)
    model["classifiers"] = nn.ModuleDict()

    for name, num_classes in zip(names, classes):
        classifier = mlp_classifier_factory(
            encoder.output_size,
            classifier_hidden_dim,
            num_classes,
            layers=classifer_num_layers,
            dropout=dropout,
        )
        model["classifiers"][name] = classifier

    return model


def build_optimizer(model, config):
    lr = config["learning_rate"]
    weight_decay = config.get("weight_decay", 0.01)

    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def save_model(
    model,
    optimizer,
    vocabulary,
    rule_encoder,
    tag_encoder,
    char2index,
    index2char,
    name,
):
    rule_encoder = {key: value for key, value in rule_encoder.items()}
    if tag_encoder is not None:
        tag_encoder = {key: value for key, value in tag_encoder.items()}

    info = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocabulary": vocabulary,
        "rule_encoder": rule_encoder,
        "tag_encoder": tag_encoder,
        "char2index": char2index,
        "index2char": index2char,
    }

    if not os.path.exists("./saved_models"):
        os.makedirs("./saved_models")

    path = os.path.join(".", "saved_models", name + ".pt")
    torch.save(info, path)
