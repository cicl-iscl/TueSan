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
from evaluate import wrong_predictions
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
    model, eval_data, eval_dataloader, indexer, device, tag_rules, translit=False
):

    eval_predictions = evaluate_model(
        model, eval_dataloader, indexer, device, tag_rules, translit
    )

    # Evaluate
    if translit:
        eval_data = convert_eval_if_translit(eval_data)

    scores = evaluate([dp[1] for dp in eval_data], eval_predictions, task_id="t2")

    # Examine wrong predictions
    print("\nWrite wrong predictions to file")
    wrong_predictions(eval_predictions, eval_dataset, predicted_rules, candidates)

    return scores


def train_model(config, checkpoint_dir=None):
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
    # test_data = load_task2_test_data(
    #    Path(config["test_path"], "task_2_input_sentences.tsv"), translit
    # )

    # logger.info(f"Loaded {len(train_data)} train sents")
    # logger.info(f"Loaded {len(eval_data)} test sents")
    # logger.info(f"Loaded {len(test_data)} test sents")

    # Generate datasets
    # logger.info("Generate training dataset")
    tag_rules = config["tag_rules"]
    stemming_rule_cutoff = config["stemming_rule_cutoff"]
    train_data, stem_rules, tags, discarded = construct_train_dataset(
        train_data, tag_rules, stemming_rule_cutoff
    )
    # logger.info(f"Training data contains {len(train_data)} sents")
    # logger.info(f"Discarded {discarded} invalid sents from train data")
    # logger.info(f"Collected {len(stem_rules)} Stemming rules")
    # logger.info(f"Collected {len(tags)} morphological tags")

    # if tag_rules:
    #    logger.info("Stemming rules contain morphological tag")
    # else:
    #    logger.info("Morphological tags are predicted separately from stems")

    # if not test:
    #    evaluate_coverage(eval_data, stem_rules, logger, tag_rules)

    # logger.info("Index dataset")

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

    # Build model
    model = build_model(config, indexer, tag_rules)
    use_cuda = config["cuda"]
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    # Build optimizer
    optimizer = build_optimizer(model, config)

    if checkpoint_dir is not None:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"), map_location=device
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        res = model, optimizer

    # Train
    epochs = config["epochs"]
    evaluator = partial(
        pred_eval,
        eval_data=eval_data,
        eval_dataloader=eval_dataloader,
        indexer=indexer,
        device=device,
        tag_rules=tag_rules,
        translit=config["translit"],
    )
    tune = config["tune"]

    if not tune:
        logger.info(f"Training for {config['epochs']} epochs")
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model has {num_parameters} trainable parameters")

    if tune or checkpoint_dir is None:
        res = train(
            model,
            optimizer,
            train_dataloader,
            epochs,
            device,
            tag_rules,
            config["lr"],
            evaluator,
            tune=tune,
            verbose=not config["tune"],
        )

    if not tune:
        return res


def main(tune, num_samples=10, max_num_epochs=20, gpus_per_trial=1):
    logger.info(f"Tune: {tune}")
    config = tune_config if tune else train_config
    config["tune"] = tune
    test = config["test"]

    start = time.time()
    device = "cuda" if torch.cuda.is_available() and config["cuda"] else "cpu"

    if tune:
        scheduler = ASHAScheduler(
            metric="score",
            mode="max",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2,
        )
        reporter = CLIReporter(
            parameter_columns=[
                "epochs",
                "embedding_size",
                "translit",
                "stemming_rule_cutoff",
                "tag_rules",
                "classifier_hidden_dim",
                "encoder_char2token_mode",
                "encoder_hidden_size",
            ],
            metric_columns=[
                "loss",
                "score",
                "training_iteration",
            ],  # we don't have accuracy now
            max_report_frequency=300,  # report every 5 min
        )

        # Tuning
        result = hyperparam_tune.run(
            partial(
                train_model,
                checkpoint_dir=config["checkpoint_dir"],
            ),
            resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
            config=config,  # our search space
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            name="T2_tune",
            log_to_file=True,
            fail_fast=True,  # stopping after first failure
            # resume=True,
        )

        best_trial = result.get_best_trial("score", "max", "last")
        logger.info(f"Best trial config: {best_trial.config}")
        best_loss = best_trial.last_result["loss"]
        logger.info(f"Best trial final validation loss: {best_loss}")

        config = best_trial.config
        with open("best_config_t2.pickle", "wb") as cf:
            pickle.dump(best_trial.config, cf)

    else:
        # model, optimizer = train_model(train_config, train_config["checkpoint_dir"])
        model, optimizer = train_model(train_config)

    duration = time.time() - start

    if test:
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

        # logger.info(f"Loaded {len(train_data)} train sents")
        # logger.info(f"Loaded {len(eval_data)} test sents")
        # logger.info(f"Loaded {len(test_data)} test sents")

        # Generate datasets
        # logger.info("Generate training dataset")
        tag_rules = config["tag_rules"]
        stemming_rule_cutoff = config["stemming_rule_cutoff"]
        train_data, stem_rules, tags, discarded = construct_train_dataset(
            train_data, tag_rules, stemming_rule_cutoff
        )
        # logger.info(f"Training data contains {len(train_data)} sents")
        # logger.info(f"Discarded {discarded} invalid sents from train data")
        # logger.info(f"Collected {len(stem_rules)} Stemming rules")
        # logger.info(f"Collected {len(tags)} morphological tags")

        # if tag_rules:
        #    logger.info("Stemming rules contain morphological tag")
        # else:
        #    logger.info("Morphological tags are predicted separately from stems")

        # if not test:
        #    evaluate_coverage(eval_data, stem_rules, logger, tag_rules)

        # logger.info("Index dataset")

        # Build vocabulary and index the dataset
        indexed_train_data, indexed_eval_data, indexer = index_dataset(
            train_data, eval_data, stem_rules, tags, tag_rules
        )

        if tune:
            best_trained_model = build_model(best_trial.config, indexer, tag_rules)
            best_trained_model.to(device)

            best_checkpoint_dir = best_trial.checkpoint.value
            model_state, optimizer_state = torch.load(
                Path(best_checkpoint_dir, "checkpoint")
            )
            best_trained_model.load_state_dict(model_state)
            model = best_trained_model

        logger.info("Evaluating on eval data")
        eval_dataloader = DataLoader(
            indexed_eval_data,
            batch_size=64,
            collate_fn=eval_collate_fn,
            shuffle=False,
        )

        scores = pred_eval(
            model, eval_data, eval_dataloader, indexer, device, tag_rules, translit
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
    parser = argparse.ArgumentParser(description="Task 1")
    parser.add_argument(
        "--tune", action="store_true", help="whether to tune hyperparams"
    )
    args = parser.parse_args()

    tune = args.tune
    main(tune, num_samples=30, max_num_epochs=20, gpus_per_trial=1)  # test
