"""
Implements helper function related to creating models, optimizer, and loss
"""


import torch
import torch.nn as nn

from segmenter_model import SegmenterModel


def build_model(config, indexer):
    # Read hyperparameters from config
    embedding_dim = config["embedding_dim"]
    hidden_dim = config["hidden_dim"]
    max_ngram = config["max_ngram"]
    dropout = config["dropout"]
    use_lstm = config["use_lstm"]

    vocabulary_size = len(indexer.vocabulary)
    num_classes = len(indexer.rules) + 1  # Add 1 for padding

    return SegmenterModel(
        vocabulary_size,
        num_classes,
        embedding_dim,
        hidden_dim,
        max_ngram,
        dropout=dropout,
        use_lstm=use_lstm,
    )


def build_optimizer(model, config):
    return torch.optim.SGD(
        model.parameters(),
        config["lr"],
        momentum=config["momentum"],
        nesterov=config["nesterov"],
        weight_decay=config["weight_decay"],
    )


def build_loss(indexer, rules, device, class_weighting=False):
    with torch.no_grad():
        class_weights = torch.zeros(len(rules) + 1, device=device)
        class_weights.required_grad = False

        for index, rule in indexer.index2rule.items():
            class_weights[index] = rules[rule]

        class_weights = class_weights / class_weights.sum()
        class_weights = 1 - class_weights

    if class_weighting:
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)

    return criterion
