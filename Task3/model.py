import os

import torch
import torch.nn as nn

from segmenter_model import Encoder, TripleHeadClassifier


def build_model(config, indexer):
    # Read hyperparameters from config
    embedding_dim = config["embedding_dim"]
    hidden_dim = config["hidden_dim"]
    max_ngram = config["max_ngram"]
    dropout = config["dropout"]

    vocabulary_size = len(indexer.vocabulary)
    num_sandhi_classes = len(indexer.sandhi_rules) + 1  # Add 1 for padding
    num_stem_classes = len(indexer.stem_rules)
    num_tag_classes = len(indexer.tags) + 1

    encoder = Encoder(
        vocabulary_size, embedding_dim, hidden_dim, max_ngram, dropout=dropout,
    )

    model = TripleHeadClassifier(
        encoder, num_sandhi_classes, num_stem_classes, num_tag_classes
    )
    return model


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters())


def build_loss(indexer, rules, device, class_weighting=False):
    with torch.no_grad():
        class_weights = torch.zeros(len(rules) + 1, device=device)
        class_weights.required_grad = False

        for index, rule in indexer.index2rule.items():
            class_weights[index] = rules[rule]

        class_weights = class_weights / class_weights.sum()
        class_weights = 1 - class_weights

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
    return criterion


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


def load_model(name, config, vocabulary):
    model = build_model(config, vocabulary)
    checkpoint = torch.load(os.path.join(".", "saved_models", name + ".pt"))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model
