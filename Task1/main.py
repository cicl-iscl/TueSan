import json
import pickle

import torch
from torch.utils.data import DataLoader

from helpers import load_data, load_sankrit_dictionary
from vocabulary import make_vocabulary

# from encode_syllables import make_syll_vocabulary
from generate_train_data import construct_dataset
from dataset import index_dataset, collate_fn, eval_collate_fn
from model import build_model, get_loss, build_optimizer, save_model, load_model
from training import train
from predicting import make_predictions
from helpers import save_task1_predictions
from scoring import evaluate

from pathlib import Path
import time
from logger import logger
import pprint

pp = pprint.PrettyPrinter(indent=4)


if __name__ == "__main__":
    with open("config.cfg") as cfg:
        config = json.load(cfg)

    translit = config["translit"]

    # Load data
    logger.info("Load data")
    train_data = load_data(config["train_path"], translit)
    eval_data = load_data(config["eval_path"], translit)

    # Make vocabulary
    # whitespaces are kept and treated as a normal character
    logger.info("Make vocab")
    vocabulary, char2index, index2char = make_vocabulary(train_data)

    logger.debug(f"{len(vocabulary)} chars in vocab:\n{vocabulary}")

    # Make syllable vocab...
    # logger.info("Collecting syllables")
    # syll_vocab, syll2index, index2syll = make_syll_vocabulary(train_data)
    # logger.debug(f"{len(syll_vocab)} unique syllables in vocab:\n{syll_vocab}")

    # # Double check vocab of dev
    # eval_vocabulary, _, _= \
    #   make_vocabulary(eval_data)
    # for char in eval_vocabulary:
    #   if char not in vocabulary:
    #       logger.warning(f"{char} not in train vocab.")

    filename = ""
    if translit:
        filename = "translit"
    else:
        filename = "unicode"

    # Construct train data, discarded 457 sents
    if Path(config["out_folder"], f"{filename}_train.pickle").is_file():
        with open(Path(config["out_folder"], f"{filename}_train.pickle"), "rb") as f:
            train_dataset = pickle.load(f)
    else:
        dat = {}
        with open(config["train_path"], encoding="utf-8") as train_json:
            dat = json.load(train_json)
        train_dataset = construct_dataset(dat, translit, config["train_graphml"])
        with open(Path(config["out_folder"], f"{filename}_train.pickle"), "wb") as outf:
            pickle.dump(train_dataset, outf)

    # Construct eval data, could have discarded 38 sentences, but didn't
    if Path(config["out_folder"], f"{filename}_dev.pickle").is_file():
        with open(Path(config["out_folder"], f"{filename}_dev.pickle"), "rb") as f:
            eval_dataset = pickle.load(f)
    else:
        eval_dat = {}
        with open(config["eval_path"], encoding="utf-8") as eval_json:
            eval_dat = json.load(eval_json)
        eval_dataset = construct_dataset(
            eval_dat, translit, config["eval_graphml"], eval=True
        )
        with open(Path(config["out_folder"], f"{filename}_dev.pickle"), "wb") as outf:
            pickle.dump(eval_dataset, outf)

    # Display an example datapoint
    pp.pprint(train_dataset[-1])
    pp.pprint(eval_dataset[-1])

    # Index data
    logger.info("Index train data")
    train_data_indexed = index_dataset(train_dataset, char2index)
    logger.info("Index eval data")
    eval_data_indexed = index_dataset(eval_dataset, char2index, eval=True)

    # Display an example
    logger.info("Example eval datapoint")
    pp.pprint(eval_dataset[-1])
    pp.pprint(eval_data_indexed[-1])

    # Build dataloaders
    batch_size = config["batch_size"]

    train_dataloader = DataLoader(
        train_data_indexed, batch_size=batch_size, collate_fn=collate_fn
    )

    eval_dataloader = DataLoader(
        eval_data_indexed,
        batch_size=batch_size,
        collate_fn=eval_collate_fn,
        shuffle=False,
    )

    # # Build model
    logger.info("Build model")
    model = build_model(config, vocabulary)
    use_cuda = config["cuda"]
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using devide: {device}")

    model = model.to(device)

    # # Build optimizer
    optimizer = build_optimizer(model)  # may need config

    # Train
    logger.info("Train\n")
    epochs = config["epochs"]
    criterion = get_loss(config)
    start = time.time()

    model, optimizer = train(
        model, criterion, optimizer, train_dataloader, epochs, device
    )

    # # # Save model
    name = config["name"]
    save_model(model, optimizer, vocabulary, char2index, index2char, name)

    # Load model
    # model = load_model(name, config, vocabulary)

    # Predictions
    logger.info("Predict\n")
    predictions, true_unsandhied, split_predictions = make_predictions(
        model, eval_dataloader, eval_dataset, device, translit
    )

    # (false) end of prediction
    duration = time.time() - start
    logger.info(f"Duration: {duration:.2f} seconds.\n")

    # Example prediction
    logger.info(f"Example prediction")
    idx = 10

    logger.info("Input sentence:")
    logger.debug(eval_dataset[idx]["sandhied"])

    logger.info("Predicted split points:")
    logger.debug(split_predictions[idx])
    logger.debug(
        [
            eval_dataset[idx]["sandhied"][i:j]
            for i, j in zip(
                [0] + split_predictions[idx], split_predictions[idx] + [None]
            )
        ]
    )

    logger.info("Predicted segmentation:")
    logger.debug(predictions[idx])
    logger.info("Gold segmentation:")
    logger.debug(true_unsandhied[idx])

    # Create submission
    logger.info("Create submission files")
    save_task1_predictions(predictions, duration)

    # Evaluation
    scores = evaluate(true_unsandhied, predictions, task_id="t1")
