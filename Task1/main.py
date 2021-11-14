import json
import pickle

import torch
from torch.utils.data import DataLoader

from helpers import load_data

from uni2intern import internal_transliteration_to_unicode as to_uni
from generate_dataset import construct_train_dataset, construct_eval_dataset
from index_dataset import index_dataset, train_collate_fn, eval_collate_fn
from model import build_model, build_optimizer, save_model, load_model, build_loss
from training import train
from predicting import make_predictions
from helpers import save_task1_predictions
from scoring import evaluate

# from pathlib import Path
import time
from logger import logger
import pprint

pp = pprint.PrettyPrinter(indent=4)

import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    with open("config.cfg") as cfg:
        config = json.load(cfg)

    translit = config["translit"]

    # Load data
    logger.info("Load data")
    train_data = load_data(config["train_path"], translit)
    eval_data = load_data(config["eval_path"], translit)

    logger.info(f"Loaded {len(train_data)} train sents")
    logger.info(f"Loaded {len(eval_data)} test sents")

    # Generate datasets
    logger.info("Generate training dataset")
    train_data, rules, discarded = construct_train_dataset(train_data)
    logger.info(f"Training data contains {len(train_data)} sents")
    logger.info(f"Collected {len(rules)} Sandhi rules")
    logger.info(f"Discarded {discarded} invalid sents from train data")

    logger.info("Generate evaluation dataset")
    eval_data = construct_eval_dataset(eval_data)

    # Build vocabulary and index the dataset
    indexed_train_data, indexed_eval_data, indexer = index_dataset(
        train_data, eval_data, rules
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

    # # Build model
    logger.info("Build model")
    model = build_model(config, indexer)
    use_cuda = config["cuda"]
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using devide: {device}")

    model = model.to(device)

    # # Build optimizer
    optimizer = build_optimizer(model)  # may need config
    criterion = build_loss(indexer, rules, device)

    # Train
    logger.info("Train\n")
    epochs = config["epochs"]
    # criterion = get_loss(config)
    start = time.time()

    model, optimizer = train(
        model, optimizer, criterion, train_dataloader, epochs, device
    )

    # Save model
    # name = config["name"]
    # save_model(model, optimizer, vocabulary, char2index, index2char, name)

    # Load model
    # model = load_model(name, config, vocabulary)

    # Predictions
    logger.info("Predict\n")
    eval_dataloader = DataLoader(
        indexed_eval_data,
        batch_size=batch_size,
        collate_fn=eval_collate_fn,
        shuffle=False,
    )

    predictions = make_predictions(
        model, eval_dataloader, indexer, device, translit=translit
    )
    predictions = [predicted.split(" ") for predicted in predictions]
    true_unsandhied = [to_uni(unsandhied).split(" ") for _, unsandhied in eval_data]

    # (false) end of prediction
    duration = time.time() - start
    logger.info(f"Duration: {duration:.2f} seconds.\n")

    # Example prediction
    logger.info(f"Example prediction")
    idx = 0

    logger.info("Input sentence:")
    logger.debug(eval_data[idx][0])

    logger.info("Predicted segmentation:")
    logger.debug(predictions[idx])
    logger.info("Gold segmentation:")
    logger.debug(true_unsandhied[idx])

    # Create submission
    logger.info("Create submission files")
    save_task1_predictions(predictions, duration)

    # Evaluation
    scores = evaluate(true_unsandhied, predictions, task_id="t1")
