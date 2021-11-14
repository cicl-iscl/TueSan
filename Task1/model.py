import os

import torch
import torch.nn as nn

from segmenter_model import SegmenterModel


def build_model(config, indexer):
    # Read hyperparameters from config
    embedding_dim = config["embedding_dim"]
    hidden_dim = config["hidden_dim"]
    max_ngram = config["max_ngram"]
    dropout = config["dropout"]

    vocabulary_size = len(indexer.vocabulary)
    num_classes = len(indexer.rules) + 1  # Add 1 for padding

    return SegmenterModel(
        vocabulary_size,
        num_classes,
        embedding_dim,
        hidden_dim,
        max_ngram,
        dropout=dropout,
    )


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters())


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
