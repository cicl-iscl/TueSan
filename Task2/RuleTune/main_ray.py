import json
import time
import torch
import pprint
import warnings

import torch.nn as nn
from pathlib import Path
from logger import logger
from training import train
from scoring import evaluate
from functools import partial
from torch.utils.data import DataLoader
from stemming_rules import evaluate_coverage
from generate_dataset import construct_train_dataset
from helpers import load_data, save_task2_predictions
from model import build_model, build_optimizer, save_model
from uni2intern import internal_transliteration_to_unicode as to_uni
from index_dataset import index_dataset, train_collate_fn, eval_collate_fn

from evaluate import (
    evaluate_model,
    print_metrics,
    format_predictions,
    convert_eval_if_translit,
)

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Ignore warning (who cares?)

warnings.filterwarnings("ignore")
pp = pprint.PrettyPrinter(indent=4)


def train_model(config, checkpoint_dir=None):
    translit = config["translit"]

    # Load data
    logger.info("Load data")
    train_data = load_data(config["train_path"], translit)
    eval_data = load_data(config["eval_path"], translit)

    logger.info(f"Loaded {len(train_data)} train sents")
    logger.info(f"Loaded {len(eval_data)} test sents")

    # Generate datasets
    logger.info("Generate training dataset")
    tag_rules = config["tag_rules"]
    stemming_rule_cutoff = config["stemming_rule_cutoff"]
    train_data, stem_rules, tags, discarded = construct_train_dataset(
        train_data, tag_rules, stemming_rule_cutoff
    )
    logger.info(f"Training data contains {len(train_data)} sents")
    logger.info(f"Discarded {discarded} invalid sents from train data")
    logger.info(f"Collected {len(stem_rules)} Stemming rules")
    logger.info(f"Collected {len(tags)} morphological tags")

    if tag_rules:
        logger.info("Stemming rules contain morphological tag")
    else:
        logger.info("Morphological tags are predicted separately from stems")

    evaluate_coverage(eval_data, stem_rules, logger, tag_rules)

    logger.info("Index dataset")

    # Build vocabulary and index the dataset
    indexed_train_data, indexed_eval_data, indexer = index_dataset(
        train_data, eval_data, stem_rules, tags, tag_rules
    )

    logger.info(f"{len(indexer.vocabulary)} chars in vocab:\n{indexer.vocabulary}\n")

    # Build dataloaders
    logger.info("Build training dataloader")
    batch_size = config["batch_size"]
    train_dataloader = DataLoader(
        indexed_train_data,
        batch_size=batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
    )

    # Build model
    logger.info("Build model")
    model = build_model(config, indexer, tag_rules)

    use_cuda = config["cuda"]
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using devide: {device}")

    model = model.to(device)

    # Build optimizer
    logger.info("Build optimizer")
    optimizer = build_optimizer(model, config)

    # Train
    epochs = config["epochs"]
    logger.info(f"Training for {epochs} epochs\n")

    start = time.time()

    train(
        model,
        optimizer,
        train_dataloader,
        epochs,
        device,
        tag_rules,
        checkpoint_dir=config["checkpoint_dir"],
    )


def main(num_samples=10, max_num_epochs=20, gpus_per_trial=1):
    # Test to see if things work as expected
    config = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        # "batch_size": tune.choice([32, 64, 128]),
        "batch_size": 64,
        "epochs": tune.choice([1, 2]),
        "momentum": 0,
        "nesterov": False,
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "embedding_size": tune.choice([32, 64, 128, 256]),
        "encoder_hidden_size": 512,
        "encoder_max_ngram": 6,
        "encoder_char2token_mode": "max",
        "classifier_hidden_dim": 512,
        "classifer_num_layers": 2,
        "dropout": 0.2,
        "tag_rules": True,
        "stemming_rule_cutoff": 5,
        "name": "test_translit",
        "translit": True,
        "train_path": "/data/jingwen/sanskrit/wsmp_train.json",
        "eval_path": "/data/jingwen/sanskrit/corrected_wsmp_dev.json",
        "cuda": True,
        "submission_dir": "result_submission",
        "checkpoint_dir": "./checkpoint",
    }

    # Define search space
    # config = {
    #     "learning_rate": tune.loguniform(1e-4, 1e-1),
    #     # "batch_size": tune.choice([32, 64, 128]),
    #     "batch_size": 64,
    #     "epochs": tune.choice([10, 15, 20, 25, 30]),
    #     "momentum": tune.choice([0, 0.9]),
    #     "nesterov": tune.choice([True, False]),
    #     "weight_decay": tune.loguniform(1e-4, 1e-1),
    #     "embedding_size": tune.choice([32, 64, 128, 256]),
    #     "encoder_hidden_size": tune.choice([32, 64, 128, 256]),
    #     "encoder_max_ngram": tune.choice([6, 7, 8, 9]),
    #     "encoder_char2token_mode": "max",
    #     "classifier_hidden_dim": tune.choice([256, 512, 1024]),
    #     "classifer_num_layers": tune.choice([1, 2, 3]),
    #     "dropout": tune.choice([0, 0.1, 0.2]),
    #     "tag_rules": tune.choice([True, False]),
    #     "stemming_rule_cutoff": 5,
    #     "name": "test_translit",
    #     "translit": tune.choice([True, False]),
    #     "train_path": "/data/jingwen/sanskrit/wsmp_train.json",
    #     "eval_path": "/data/jingwen/sanskrit/corrected_wsmp_dev.json",
    #     "cuda": True,
    #     "submission_dir": "result_submission",
    #     "checkpoint_dir": "./checkpoint",
    # }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    reporter = CLIReporter(
        parameter_columns=[
            "epochs",
            "learning_rate",
            "embedding_dim",
            "encoder_max_ngram",
        ],
        metric_columns=[
            "loss",
            "accuracy",
            "training_iteration",
        ],  # we don't have accuracy now
        max_report_frequency=300,  # report every 5 min
    )

    # ================REPETITIVE CODE==========================
    translit = config["translit"]
    tag_rules = config["tag_rules"]

    # Load data
    logger.info("Load data again...")
    train_data = load_data(config["train_path"], translit)
    eval_data = load_data(config["eval_path"], translit)

    # Generate datasets
    logger.info("Generate training dataset again...")
    tag_rules = config["tag_rules"]
    stemming_rule_cutoff = config["stemming_rule_cutoff"]
    train_data, stem_rules, tags, discarded = construct_train_dataset(
        train_data, tag_rules, stemming_rule_cutoff
    )

    if tag_rules:
        logger.info("Stemming rules contain morphological tag")
    else:
        logger.info("Morphological tags are predicted separately from stems")

    logger.info("Index dataset again...")

    # Build vocabulary and index the dataset
    indexed_train_data, indexed_eval_data, indexer = index_dataset(
        train_data, eval_data, stem_rules, tags, tag_rules
    )
    # Build eval dataloader
    batch_size = config["batch_size"]
    eval_dataloader = DataLoader(
        indexed_eval_data,
        batch_size=batch_size,
        collate_fn=eval_collate_fn,
        shuffle=False,
    )

    # ================END REPETITIVE CODE==========================

    start = time.time()

    # Tuning
    result = tune.run(
        partial(
            train_model,
            checkpoint_dir=config["checkpoint_dir"],
        ),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,  # our search space
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="T1_tune",
        log_to_file=True,
        fail_fast=True,  # stopping after first failure
        # resume=True,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    # ---- Have to tune.report accuracy ! ----
    # print(
    #     "Best trial final validation accuracy: {}".format(
    #         best_trial.last_result["accuracy"]
    #     )
    # )

    best_trained_model = build_model(best_trial.config, indexer, tag_rules)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        # if gpus_per_trial > 1:
        #     best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(Path(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    pred_eval(
        best_trained_model,
        eval_data,
        eval_dataloader,
        indexer,
        device,
        tag_rules,
        start,
        translit=translit,
    )


def pred_eval(
    model, eval_data, eval_dataloader, indexer, device, tag_rules, start, translit=False
):

    eval_predictions = evaluate_model(
        model, eval_dataloader, indexer, device, tag_rules, translit
    )

    duration = time.time() - start
    logger.info(f"Duration: {duration:.2f} seconds.\n")

    # Create submission
    logger.info("Create submission files")
    save_task2_predictions(eval_predictions, duration)

    # Evaluate
    logger.info("Evaluating")
    # print_metrics(eval_predictions, eval_dataset)
    # Task 2 Evaluation
    if translit:
        eval_data = convert_eval_if_translit(eval_data)
    scores = evaluate([dp[1] for dp in eval_data], eval_predictions, task_id="t2")


if __name__ == "__main__":
    main(num_samples=2, max_num_epochs=20, gpus_per_trial=1)  # test
    # main(num_samples=20, max_num_epochs=25, gpus_per_trial=1)
