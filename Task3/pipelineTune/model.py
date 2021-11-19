import os

import torch
import torch.nn as nn

from segmenter_model import SegmenterModel
from classifier import Classifier


def build_model(config, indexer):
    # Read hyperparameters from config
    embedding_dim = config["embedding_dim"]
    hidden_dim = config["hidden_dim"]
    max_ngram = config["max_ngram"]
    dropout = config["dropout"]
    mode = config["char2token_mode"]

    vocabulary_size = len(indexer.vocabulary)
    num_sandhi_classes = len(indexer.sandhi_rules) + 1  # Add 1 for padding
    num_stem_classes = len(indexer.stem_rules) + 1
    num_tag_classes = len(indexer.tags)

    segmenter = SegmenterModel(
        vocabulary_size,
        embedding_dim,
        hidden_dim,
        max_ngram,
        num_sandhi_classes,
        dropout=dropout,
    )

    classifier = Classifier(
        hidden_dim,
        num_stem_classes,
        num_tag_classes,
        char2token_mode=mode,
        dropout=dropout,
    )

    model = nn.ModuleDict({"segmenter": segmenter, "classifier": classifier})
    return model


def build_optimizer(model, config):
    return torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        momentum=config["momentum"],
        nesterov=config["nesterov"],
    )


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
