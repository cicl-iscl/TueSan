import time
import json
import torch
import pprint
import pickle
import warnings

from logger import logger
from training import train
from model import load_model
from model import save_model
from scoring import evaluate
from helpers import load_data
from model import build_model
from model import build_optimizer
from torch.utils.data import DataLoader
from predicting import make_predictions
from index_dataset import index_dataset
from index_dataset import eval_collate_fn
from index_dataset import train_collate_fn
from helpers import save_task3_predictions
from stemming_rules import evaluate_coverage
from generate_dataset import construct_train_dataset
from uni2intern import internal_transliteration_to_unicode as to_uni


pp = pprint.PrettyPrinter(indent=4)
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

    # # Build model
    logger.info("Build model")
    model = build_model(config, indexer)
    use_cuda = config["cuda"]
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using devide: {device}")

    model = model.to(device)

    # Build optimizer
    optimizer = build_optimizer(model)  # may need config

    # Train
    logger.info("Train\n")
    epochs = config["epochs"]
    start = time.time()

    model, optimizer = train(model, optimizer, train_dataloader, epochs, device)

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
    # predictions = [predicted.split(" ") for predicted in predictions]
    # true_unsandhied = [to_uni(unsandhied).split(" ") for _, unsandhied in eval_data]

    # (false) end of prediction
    duration = time.time() - start
    logger.info(f"Duration: {duration:.2f} seconds.\n")

    # Example prediction
    logger.info(f"Example prediction")
    idx = 0

    logger.info("Input sentence:")
    logger.debug(eval_data[idx])

    logger.info("Predicted segmentation:")
    logger.debug(predictions[idx])
    # logger.info("Gold segmentation:")
    # logger.debug(true_unsandhied[idx])

    # Create submission
    logger.info("Create submission files")
    save_task3_predictions(predictions, duration)

    # Evaluation
    ground_truth = [labels for _, labels in eval_data]
    scores = evaluate(ground_truth, predictions, task_id="t3")
