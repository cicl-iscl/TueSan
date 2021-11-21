import time
import json
import pickle
import argparse
from pathlib import Path
import os

from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ray import tune as hyperparam_tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from helpers import load_data, load_task1_test_data
from generate_dataset import construct_train_dataset, construct_eval_dataset
from index_dataset import index_dataset, train_collate_fn, eval_collate_fn
from model import build_model, build_optimizer, build_loss
from training import train
from predicting import make_predictions
from uni2intern import internal_transliteration_to_unicode as to_uni
from helpers import save_task1_predictions
from scoring import evaluate, print_scores

import pickle
from config import train_config, tune_config


from logger import logger
import pprint

pp = pprint.PrettyPrinter(indent=4)

import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(12345)


def pred_eval(
    model,
    eval_data,
    eval_dataloader,
    indexer,
    device,
    translit=False,
    verbose=False,
):
    predictions = make_predictions(
        model, eval_dataloader, indexer, device, translit=translit
    )
    if translit:
        predictions = [
            to_uni(predicted).split(" ") for predicted in predictions
        ]
        true_unsandhied = [
            to_uni(unsandhied).split(" ") for _, unsandhied in eval_data
        ]
    else:
        predictions = [predicted.split(" ") for predicted in predictions]
        true_unsandhied = [
            unsandhied.split(" ") for _, unsandhied in eval_data
        ]

    # ----- Example prediction -------
    if verbose:
        logger.info(f"Example prediction")
        idx = 0

        logger.info("Input sentence:")
        logger.debug(eval_data[idx][0])

        logger.info("Predicted segmentation:")
        logger.debug(predictions[idx])
        logger.info("Gold segmentation:")
        logger.debug(true_unsandhied[idx])
        # ---------------------------------

    # Evaluation
    scores = evaluate(true_unsandhied, predictions, task_id="t1")
    return scores


def evaluate_checkpoint(config, checkpoint, checkpoint_dir=None):

    translit = config["translit"]

    # Load data
    # logger.info("Load data")
    train_data = load_data(config["train_path"], translit)
    eval_data = load_data(config["eval_path"], translit)
    test_data = load_task1_test_data(
        Path(config["test_path"], "task_1_input_sentences.tsv"), translit
    )

    # Generate datasets
    # logger.info("Generate training dataset")
    train_data, rules, discarded = construct_train_dataset(train_data)

    # logger.info("Generate evaluation dataset")
    eval_data = construct_eval_dataset(eval_data)

    # Build vocabulary and index the dataset
    indexed_train_data, indexed_eval_data, indexer = index_dataset(
        train_data, eval_data, rules
    )

    # logger.info(f"{len(indexer.vocabulary)} chars in vocab:\n{indexer.vocabulary}\n")

    # Build dataloaders
    # logger.info("Build training dataloader")
    batch_size = config["batch_size"]
    train_dataloader = DataLoader(
        indexed_train_data,
        batch_size=batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
    )

    eval_dataloader = DataLoader(
        indexed_eval_data,
        batch_size=batch_size,
        collate_fn=eval_collate_fn,
        shuffle=False,
    )

    # Read best config
    with open("best_config_t2.pickle", "rb") as cf:
        best_config = pickle.load(best_trial.config, cf)

    # Build model
    # logger.info("Build model")
    model = build_model(best_config, indexer)
    use_cuda = best_config["cuda"]
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    # Build optimizer
    optimizer = build_optimizer(model, best_config)  # may need config
    criterion = build_loss(indexer, rules, device)  # may need config

    if checkpoint_dir is not None:
        model_state, optimizer_state = torch.load(
            Path(checkpoint_dir, checkpoint)
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

        logger.info("Evaluating on eval data")
        eval_dataloader = DataLoader(
            indexed_eval_data,
            batch_size=64,
            collate_fn=eval_collate_fn,
            shuffle=False,
        )
        scores = pred_eval(
            model,
            eval_data,
            eval_dataloader,
            indexer,
            device,
            translit=translit,
        )
        print_scores(scores)

        logger.info("Creating predictions on test data")
        test_data = [sent for sent, _ in test_data]
        indexed_test_data = list(map(indexer.index_sent, test_data))

        test_dataloader = DataLoader(
            indexed_test_data,
            batch_size=64,
            collate_fn=eval_collate_fn,
            shuffle=False,
        )

        device = (
            "cuda" if torch.cuda.is_available() and config["cuda"] else "cpu"
        )

        predictions = make_predictions(
            model, test_dataloader, indexer, device, translit=translit
        )

        if translit:
            predictions = [
                to_uni(predicted).split(" ") for predicted in predictions
            ]

        save_task1_predictions(predictions, duration)


if __name__ == "__main__":
    checkpoint = "DEFAULT_36a06_00009_9_batch_size=16,dropout=0.0,embedding_dim=128,epochs=20,hidden_dim=512,translit=True,use_lstm=True_2021-11-21_00-05-10/checkpoint_000015/checkpoint"
    evaluate_checkpoint(
        config, checkpoint=checkpoint, checkpoint_dir="~/ray_results/T1_tune"
    )
