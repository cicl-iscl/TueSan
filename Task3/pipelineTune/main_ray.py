import json
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial

from helpers import load_data

from uni2intern import internal_transliteration_to_unicode as to_uni
from generate_dataset import construct_train_dataset
from index_dataset import index_dataset, train_collate_fn, eval_collate_fn
from model import build_model, build_optimizer, save_model, load_model
from training import train
from predicting import make_predictions
from helpers import save_task3_predictions
from scoring import evaluate
from stemming_rules import evaluate_coverage

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from pathlib import Path
import time
from logger import logger
import pprint

pp = pprint.PrettyPrinter(indent=4)

import warnings

warnings.filterwarnings("ignore")


def train_model(
    config,
    checkpoint_dir=None,
):

    translit = config["translit"]

    # Load data
    logger.info("Load data")
    train_data = load_data(config["train_path"], translit)
    eval_data = load_data(config["eval_path"], translit)

    logger.info(f"Loaded {len(train_data)} train sents")
    logger.info(f"Loaded {len(eval_data)} test sents")

    # Generate datasets
    logger.info("Generate training dataset")
    train_data, sandhi_rules, stem_rules, tags, discarded = construct_train_dataset(
        train_data
    )
    logger.info(f"Training data contains {len(train_data)} sents")
    logger.info(f"Collected {len(sandhi_rules)} Sandhi rules")
    logger.info(f"Collected {len(stem_rules)} Stemming rules")
    logger.info(f"Collected {len(tags)} morphological tags")
    logger.info(f"Discarded {discarded} invalid sents from train data")

    evaluate_coverage(eval_data, stem_rules, logger)

    # logger.info("Stem rules")
    # for (t_pre, t_suf), (s_pre, s_suf) in stem_rules:
    #    logger.info(f"{t_pre}, {t_suf} --> {s_pre}, {s_suf}")

    logger.info("Generate evaluation dataset")

    # Build vocabulary and index the dataset
    indexed_train_data, indexed_eval_data, indexer = index_dataset(
        train_data, eval_data, sandhi_rules, stem_rules, tags
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

    eval_dataloader = DataLoader(
        indexed_eval_data,
        batch_size=batch_size,
        collate_fn=eval_collate_fn,
        shuffle=False,
    )

    # Build model
    logger.info("Build model")
    model = build_model(config, indexer)
    use_cuda = config["cuda"]
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using devide: {device}")
    # if use_cuda and torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model = model.to(device)

    # Build optimizer
    optimizer = build_optimizer(model, config)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            Path(checkpoint_dir, "checkpoint").mkdir(parents=True, exist_ok=True)
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Train
    logger.info("Train\n")
    epochs = config["epochs"]
    # criterion = get_loss(config)
    start = time.time()

    train(model, optimizer, train_dataloader, epochs, device)


def main(num_samples=10, max_num_epochs=25, gpus_per_trial=1):
    # Test to see if things work as expected
    # config = {
    #     "lr": tune.loguniform(1e-4, 1e-1),
    #     # "batch_size": tune.choice([32, 64, 128]),
    #     "batch_size": 64,
    #     "epochs": tune.choice([1, 3]),
    #     "weight_decay": tune.loguniform(1e-4, 1e-1),
    #     "momentum": tune.choice([0, 0.9]),
    #     "nesterov": False,
    #     "max_ngram": tune.choice([6, 7, 8, 9]),
    #     "dropout": tune.choice([0, 0.1, 0.2, 0.3]),
    #     "hidden_dim": tune.choice([256, 512, 1024]),
    #     "embedding_dim": tune.choice([32, 64, 128, 256]),
    #     "name": "test_translit",
    #     "translit": True,
    #     "train_path": "/data/jingwen/sanskrit/wsmp_train.json",
    #     "eval_path": "/data/jingwen/sanskrit/corrected_wsmp_dev.json",
    #     "train_graphml": "/data/jingwen/sanskrit/final_graphml_train",
    #     "eval_graphml": "/data/jingwen/sanskrit/graphml_dev",
    #     "char2token_mode": "max",
    #     "cuda": True,
    #     "dictionary_path": "/data/jingwen/sanskrit/dictionary.pickle",
    #     "out_folder": "../sanskrit",
    #     "submission_dir": "result_submission",
    #     "checkpoint_dir": "./checkpoint",
    # }
    # Define search space
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        # "batch_size": tune.choice([32, 64, 128]),
        "batch_size": 64,
        "epochs": tune.choice([10, 15, 20, 25]),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "momentum": tune.choice([0, 0.9]),
        "nesterov": False,
        "max_ngram": tune.choice([6, 7, 8]),
        "dropout": tune.choice([0, 0.1, 0.2]),
        "hidden_dim": tune.choice([256, 512, 1024]),
        "embedding_dim": tune.choice([32, 64, 128, 256]),
        "name": "test_translit",
        "translit": True,
        "train_path": "/data/jingwen/sanskrit/wsmp_train.json",
        "eval_path": "/data/jingwen/sanskrit/corrected_wsmp_dev.json",
        "train_graphml": "/data/jingwen/sanskrit/final_graphml_train",
        "eval_graphml": "/data/jingwen/sanskrit/graphml_dev",
        "char2token_mode": "max",
        "cuda": True,
        "dictionary_path": "/data/jingwen/sanskrit/dictionary.pickle",
        "out_folder": "../sanskrit",
        "submission_dir": "result_submission",
        "checkpoint_dir": "./checkpoint",
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    reporter = CLIReporter(
        parameter_columns=["epochs", "lr", "max_ngram", "hidden_dim", "embedding_dim"],
        metric_columns=["loss", "accuracy", "training_iteration"],
        max_report_frequency=300,  # report every 5 min
    )

    # ===================REPETITIVE CODE============================
    translit = config["translit"]
    # Load data
    logger.info("Load data again...")
    train_data = load_data(config["train_path"], translit)
    eval_data = load_data(config["eval_path"], translit)

    # Generate datasets
    logger.info("Generate training dataset again...")
    train_data, sandhi_rules, stem_rules, tags, discarded = construct_train_dataset(
        train_data
    )

    logger.info("Generate evaluation dataset again...")

    # Build vocabulary and index the dataset
    indexed_train_data, indexed_eval_data, indexer = index_dataset(
        train_data, eval_data, sandhi_rules, stem_rules, tags
    )

    # Build dataloaders
    logger.info("Build training dataloader again...")
    batch_size = config["batch_size"]

    eval_dataloader = DataLoader(
        indexed_eval_data,
        batch_size=batch_size,
        collate_fn=eval_collate_fn,
        shuffle=False,
    )
    # ===================REPETITIVE CODE============================

    start = time.time()
    # Tuning
    result = tune.run(
        partial(
            train_model,
            checkpoint_dir=config["checkpoint_dir"],
        ),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="T3_tune",
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

    best_trained_model = build_model(best_trial.config, indexer)
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
        start,
        translit=translit,
    )


def pred_eval(
    model, eval_data, eval_dataloader, indexer, device, start, translit=False
):

    predictions = make_predictions(
        model, eval_dataloader, indexer, device, translit=translit
    )

    duration = time.time() - start
    logger.info(f"Duration: {duration:.2f} seconds.\n")

    # -----Example prediction-----
    logger.info(f"Example prediction")
    idx = 0

    logger.info("Input sentence:")
    logger.debug(eval_data[idx])

    logger.info("Predicted segmentation:")
    logger.debug(predictions[idx])
    # logger.info("Gold segmentation:")
    # logger.debug(true_unsandhied[idx])
    # ------------------------------

    # Create submission
    logger.info("Create submission files")
    save_task3_predictions(predictions, duration)

    # Evaluation
    ground_truth = []
    if translit:
        ground_truth = []
        for _, labels in eval_data:
            translit_sent = []
            for token, stem, tag in labels:
                translit_sent.append([to_uni(token), to_uni(stem), tag])
            ground_truth.append(translit_sent)

    else:
        ground_truth = [labels for _, labels in eval_data]
    scores = evaluate(ground_truth, predictions, task_id="t3")


if __name__ == "__main__":
    # main(num_samples=2, max_num_epochs=20, gpus_per_trial=1)  # test
    main(num_samples=20, max_num_epochs=25, gpus_per_trial=1)
