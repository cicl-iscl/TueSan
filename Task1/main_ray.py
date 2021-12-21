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
from predicting import make_predictions, wrong_predictions, output_wrong_predictions
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
    model, eval_data, eval_dataloader, indexer, device, translit=False, verbose=False
):
    predictions = make_predictions(
        model, eval_dataloader, indexer, device, translit=translit
    )
    if translit:
        predictions = [to_uni(predicted).split(" ") for predicted in predictions]
        true_unsandhied = [to_uni(unsandhied).split(" ") for _, unsandhied in eval_data]
    else:
        predictions = [predicted.split(" ") for predicted in predictions]
        true_unsandhied = [unsandhied.split(" ") for _, unsandhied in eval_data]

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

    # Check predictions
    _, wrong, rules_used = wrong_predictions(
        model, eval_dataloader, indexer, device, true_unsandhied, translit=translit
    )
    output_wrong_predictions(wrong, rules_used)

    return scores


def train_model(config, checkpoint_dir=None):

    translit = config["translit"]
    # test = config["test"]

    # if translit:
    #     logger.info("Transliterating input")
    # else:
    #     logger.info("Using raw input")

    # Load data
    # logger.info("Load data")
    train_data = load_data(config["train_path"], translit)
    eval_data = load_data(config["eval_path"], translit)

    # logger.info(f"Loaded {len(train_data)} train sents")
    # logger.info(f"Loaded {len(eval_data)} eval sents")
    # logger.info(f"Loaded {len(test_data)} test sents")

    # Generate datasets
    # logger.info("Generate training dataset")
    train_data, rules, discarded = construct_train_dataset(train_data)
    # logger.info(f"Training data contains {len(train_data)} sents")
    # logger.info(f"Collected {len(rules)} Sandhi rules")
    # logger.info(f"Discarded {discarded} invalid sents from train data")

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

    # Build model
    # logger.info("Build model")
    model = build_model(config, indexer)
    use_cuda = config["cuda"]
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    # Build optimizer
    optimizer = build_optimizer(model, config)  # may need config
    criterion = build_loss(indexer, rules, device)  # may need config

    if checkpoint_dir is not None:
        model_state, optimizer_state = torch.load(
            Path(checkpoint_dir, "checkpoint").mkdir(parents=True, exist_ok=True)
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Train

    evaluator = partial(
        pred_eval,
        eval_data=eval_data,
        eval_dataloader=eval_dataloader,
        indexer=indexer,
        device=device,
        translit=config["translit"],
    )
    tune = config["tune"]

    if not tune:
        logger.info(f"Training for {config['epochs']} epochs")

    res = train(
        model,
        optimizer,
        criterion,
        train_dataloader,
        config["epochs"],
        device,
        config["lr"],
        evaluator,
        tune=tune,
        # config = config["checkpoint_dir"],
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
                "use_lstm",
                "translit",
                "epochs",
                "batch_size",
                "dropout",
                "hidden_dim",
                "embedding_dim",
            ],
            metric_columns=["loss", "score", "training_iteration"],
            max_report_frequency=300,  # report every 5 min
        )

        # Tuning
        result = hyperparam_tune.run(
            partial(train_model, checkpoint_dir=config["checkpoint_dir"]),
            resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
            config=config,  # our search space
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            name="T1_tune_test",
            log_to_file=True,
            # fail_fast=True,  # stopping after first failure
            # resume=True
        )
        best_trial = result.get_best_trial("score", "max", "last")
        logger.info("Best trial config: {}".format(best_trial.config))
        logger.info(
            "Best trial final validation loss: {}".format(
                best_trial.last_result["loss"]
            )
        )
        logger.info(
            "Best trial final task score: {}".format(best_trial.last_result["score"])
        )
        # best_trained_model = build_model(best_trial.config, indexer)
        config = best_trial.config
        with open("best_config_t1.pickle", "wb") as cf:
            pickle.dump(best_trial.config, cf)

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # best_trained_model.to(device)

        # best_checkpoint_dir = best_trial.checkpoint.value
        # model_state, optimizer_state = torch.load(
        #     Path(best_checkpoint_dir, "checkpoint")
        # )
        # best_trained_model.load_state_dict(model_state)
        # model = best_trained_model
        # config = best_trial.config

    else:
        model, optimizer = train_model(train_config)

    # (false) end of prediction
    duration = time.time() - start
    logger.info(f"Duration: {duration:.2f} seconds.\n")
    device = "cuda" if torch.cuda.is_available() and config["cuda"] else "cpu"

    if test:

        # ----------------data loading-------------------

        # Load data
        logger.info("Load data")
        translit = config["translit"]
        test = config["test"]

        # if translit:
        #     logger.info("Transliterating input")
        # else:
        #     logger.info("Using raw input")

        train_data = load_data(config["train_path"], translit)
        eval_data = load_data(config["eval_path"], translit)
        test_data = load_task1_test_data(
            Path(config["test_path"], "task_1_input_sentences.tsv"), translit
        )

        # logger.info(f"Loaded {len(train_data)} train sents")
        # logger.info(f"Loaded {len(eval_data)} eval sents")
        # logger.info(f"Loaded {len(test_data)} test sents")

        # Generate datasets
        # logger.info("Generate training dataset")
        train_data, rules, discarded = construct_train_dataset(train_data)
        # logger.info(f"Training data contains {len(train_data)} sents")
        # logger.info(f"Collected {len(rules)} Sandhi rules")
        # logger.info(f"Discarded {discarded} invalid sents from train data")

        # logger.info("Generate evaluation dataset")
        eval_data = construct_eval_dataset(eval_data)

        # Build vocabulary and index the dataset
        indexed_train_data, indexed_eval_data, indexer = index_dataset(
            train_data, eval_data, rules
        )
        # logger.info(
        #     f"{len(indexer.vocabulary)} chars in vocab:\n{indexer.vocabulary}\n"
        # )
        # -----------------------------------------------

        if tune:
            best_trained_model = build_model(best_trial.config, indexer)
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
            model, eval_data, eval_dataloader, indexer, device, translit=translit
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

        device = "cuda" if torch.cuda.is_available() and config["cuda"] else "cpu"

        predictions = make_predictions(
            model, test_dataloader, indexer, device, translit=translit
        )

        if translit:
            predictions = [to_uni(predicted).split(" ") for predicted in predictions]

        save_task1_predictions(predictions, duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1")
    parser.add_argument(
        "--tune", action="store_true", help="whether to tune hyperparams"
    )
    args = parser.parse_args()

    tune = args.tune
    # main(tune, num_samples=2, max_num_epochs=20, gpus_per_trial=1)  # test
    main(tune, num_samples=20, max_num_epochs=25, gpus_per_trial=1)
