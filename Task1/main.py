import json
import time
import torch
import pickle
import pprint
import warnings

from pathlib import Path
from logger import logger
from training import train
from scoring import evaluate
from model import save_model
from model import load_model
from model import build_loss
from helpers import load_data
from model import build_model
from model import build_optimizer
from predicting import make_predictions
from torch.utils.data import DataLoader
from index_dataset import index_dataset
from helpers import load_task1_test_data
from index_dataset import eval_collate_fn
from index_dataset import train_collate_fn
from helpers import save_task1_predictions
from generate_dataset import construct_eval_dataset
from generate_dataset import construct_train_dataset
from uni2intern import internal_transliteration_to_unicode as to_uni


pp = pprint.PrettyPrinter(indent=4)
warnings.filterwarnings("ignore")
torch.manual_seed(12345)


if __name__ == "__main__":
    with open("config.cfg") as cfg:
        config = json.load(cfg)

    # Read booleans
    translit = config["translit"]
    test = config["test"]
    tune = config["tune"]

    # Load data
    logger.info("Load data")
    train_data = load_data(config["train_path"], translit)
    eval_data = load_data(config["eval_path"], translit)
    test_data = load_task1_test_data(
        Path(config["test_path"], "task_1_input_sentences.tsv"), translit
    )

    logger.info(f"Loaded {len(train_data)} train sents")
    logger.info(f"Loaded {len(eval_data)} eval sents")
    logger.info(f"Loaded {len(test_data)} test sents")

    # Check test sents
    # logger.debug(test_data[0])
    # logger.debug(test_data[-1])

    # Generate datasets
    logger.info("Generate training dataset")
    train_data, rules, discarded = construct_train_dataset(train_data)
    logger.info(f"Training data contains {len(train_data)} sents")
    logger.info(f"Collected {len(rules)} Sandhi rules")
    logger.info(f"Discarded {discarded} invalid sents from train data")

    logger.info("Generate evaluation dataset")
    eval_data = construct_eval_dataset(eval_data)
    if test:
        eval_data = construct_eval_dataset(test_data)

    # test
    logger.info("Example test data")
    logger.debug(test_data[0])

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

    # Build model
    logger.info("Build model")
    model = build_model(config, indexer)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_parameters} trainable parameters")

    use_cuda = config["cuda"]
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using devide: {device}")

    model = model.to(device)

    # Build optimizer
    optimizer = build_optimizer(model)  # may need config
    criterion = build_loss(indexer, rules, device)

    # Train
    logger.info("Train\n")
    epochs = config["epochs"]
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
    if translit:
        predictions = [to_uni(predicted).split(" ") for predicted in predictions]
        if not test:
            true_unsandhied = [
                to_uni(unsandhied).split(" ") for _, unsandhied in eval_data
            ]
    else:
        predictions = [predicted.split(" ") for predicted in predictions]
        if not test:
            true_unsandhied = [unsandhied.split(" ") for _, unsandhied in eval_data]

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
    if not test:
        logger.info("Gold segmentation:")
        logger.debug(true_unsandhied[idx])

    # Create submission
    logger.info("Create submission files")
    save_task1_predictions(predictions, duration)

    # Evaluation
    if not test:
        scores = evaluate(true_unsandhied, predictions, task_id="t1")
