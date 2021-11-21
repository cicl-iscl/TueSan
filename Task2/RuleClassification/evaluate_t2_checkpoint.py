import os
import json
import time
import torch
import pprint
import pickle
import warnings
import argparse

import torch.nn as nn
from pathlib import Path
from logger import logger
from training import train
from scoring import evaluate
from functools import partial
from config import tune_config
from config import train_config
from ray.tune import CLIReporter
from scoring import print_scores
from evaluate import evaluate_model
from ray import tune as hyperparam_tune
from torch.utils.data import DataLoader
from helpers import load_task2_test_data
from stemming_rules import evaluate_coverage
from ray.tune.schedulers import ASHAScheduler
from evaluate import convert_eval_if_translit
from generate_dataset import construct_train_dataset
from helpers import load_data, save_task2_predictions
from model import build_model, build_optimizer, save_model
from uni2intern import internal_transliteration_to_unicode as to_uni
from index_dataset import index_dataset, train_collate_fn, eval_collate_fn

# Ignore warning (who cares?)
warnings.filterwarnings("ignore")
pp = pprint.PrettyPrinter(indent=4)
torch.manual_seed(12345)


def pred_eval(
    model,
    eval_data,
    eval_dataloader,
    indexer,
    device,
    tag_rules,
    translit=False,
):

    eval_predictions = evaluate_model(
        model, eval_dataloader, indexer, device, tag_rules, translit
    )

    # Evaluate
    if translit:
        eval_data = convert_eval_if_translit(eval_data)

    scores = evaluate(
        [dp[1] for dp in eval_data], eval_predictions, task_id="t2"
    )
    return scores


def evaluate_checkpoint(config, checkpoint, checkpoint_dir=None):
    # Load data
    # logger.info("Load data")
    translit = config["translit"]
    test = config["test"]

    # if translit:
    #    logger.info("Transliterating input")
    # else:
    #    logger.info("Using raw input")

    train_data = load_data(config["train_path"], translit)
    eval_data = load_data(config["eval_path"], translit)
    test_data = load_task2_test_data(
        Path(config["test_path"], "task_2_input_sentences.tsv"), translit
    )

    # Generate datasets
    # logger.info("Generate training dataset")
    tag_rules = config["tag_rules"]
    stemming_rule_cutoff = config["stemming_rule_cutoff"]
    train_data, stem_rules, tags, discarded = construct_train_dataset(
        train_data, tag_rules, stemming_rule_cutoff
    )

    # Build vocabulary and index the dataset
    indexed_train_data, indexed_eval_data, indexer = index_dataset(
        train_data, eval_data, stem_rules, tags, tag_rules
    )

    # logger.info(f"{len(indexer.vocabulary)} chars in vocab:\n{indexer.vocabulary}\n")

    # Build dataloaders
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
    model = build_model(best_config, indexer, tag_rules)
    use_cuda = best_config["cuda"]
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    # Build optimizer
    optimizer = build_optimizer(model, best_config)

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
            tag_rules,
            translit,
        )
        print_scores(scores)

        logger.info("Creating predictions on test data")
        # Index test data
        indexed_test_data = []
        for raw_tokens, *_ in test_data:
            raw_tokens = raw_tokens.split()
            indexed_tokens = list(map(indexer.index_token, raw_tokens))
            indexed_test_data.append((raw_tokens, indexed_tokens))

        # Create dataloader
        test_dataloader = DataLoader(
            indexed_test_data,
            batch_size=64,
            collate_fn=eval_collate_fn,
            shuffle=False,
        )

        # Get predictions
        predictions = evaluate_model(
            model, test_dataloader, indexer, device, tag_rules, translit
        )

        # Create submission
        logger.info("Create submission files")
        save_task2_predictions(predictions, duration)


if __name__ == "__main__":
    evaluate_checkpoint(
        config, checkpoint="some name", checkpoint_dir="~/ray_results/T2_tune"
    )
