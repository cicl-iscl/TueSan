"""
Helpers for building & saving model and optimizer
"""

import os
import torch

from mlp import mlp
from encoder import Encoder


def build_model(config, indexer, tag_rules):
    # Read hyperparameters from config
    embedding_size = config["embedding_size"]
    encoder_hidden_size = config["encoder_hidden_size"]
    encoder_max_ngram = config["encoder_max_ngram"]
    encoder_char2token_mode = config["encoder_char2token_mode"]
    classifier_hidden_dim = config["classifier_hidden_dim"]
    classifer_num_layers = config["classifer_num_layers"]
    dropout = config["dropout"]

    model = dict()

    # Make encoder, stem decoder, and tag classifier
    # Encoder
    encoder = Encoder(
        len(indexer.vocabulary),
        embedding_size,
        encoder_hidden_size,
        encoder_max_ngram,
        char2token_mode=encoder_char2token_mode,
        dropout=dropout,
    )
    model["encoder"] = encoder

    stem_rule_classifier = mlp(
        encoder_hidden_size,
        classifier_hidden_dim,
        classifer_num_layers,
        output_dim=len(indexer.stem_rules) + 1,
        dropout=dropout,
    )
    model["stem_rule_classifier"] = stem_rule_classifier

    if not tag_rules:
        tag_classifier = mlp(
            encoder_hidden_size,
            classifier_hidden_dim,
            classifer_num_layers,
            output_dim=len(indexer.tags),
            dropout=dropout,
        )
        model["tag_classifier"] = tag_classifier

    model = torch.nn.ModuleDict(model)

    return model


def build_optimizer(model, config):
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    momentum = config["momentum"]
    nesterov = config["nesterov"]

    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
    )

    return optimizer


def save_model(
    model, optimizer, indexer, stem_rules, tags, name,
):

    info = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "indexer": indexer,
        "stem_rules": stem_rules,
        "tags": tags,
    }

    if not os.path.exists("./saved_models"):
        os.makedirs("./saved_models")

    path = os.path.join(".", "saved_models", name + ".pt")
    torch.save(info, path)
