import json
import pickle

import torch
from torch.utils.data import DataLoader

from helpers import load_data, load_sankrit_dictionary, check_specific_sent
from vocabulary import make_vocabulary
from generate_train_data import construct_dataset
from dataset import index_dataset, collate_fn, eval_collate_fn
from model import build_model, get_loss, build_optimizer, save_model
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

    # Load data, data is a dictionary indexed by sent_id
    logger.info("Load data")
    train_data = load_data(config["train_path"], translit)
    eval_data = load_data(config["eval_path"], translit)

    # Display an example sentence
    sent = check_specific_sent(eval_data, 582923)
    pp.pprint(sent)

    # Make vocabulary
    # whitespaces are translated to '_' and treated as a normal character
    logger.info("Make vocab")
    vocabulary, char2index, index2char, char2uni = make_vocabulary(train_data.values())

    logger.debug(f"{len(vocabulary)} chars in vocab:\n{vocabulary}")

    # # Double check vocab of dev
    # eval_vocabulary, _, _, _ = \
    #   make_vocabulary(eval_data.values())
    # for char in eval_vocabulary:
    #   if char not in vocabulary:
    #       logger.warning(f"{char} not in train vocab.")

    # Construct train data, discarded 457 sents
    if Path(config["out_folder"], "translit_train.pickle").is_file():
        with open(Path(config["out_folder"], "translit_train.pickle"), "rb") as f:
            train_dataset = pickle.load(f)
    else:
        dat = {}
        with open(config["train_path"], encoding="utf-8") as train_json:
            dat = json.load(train_json)
        train_dataset = construct_dataset(dat, translit, config["train_graphml"])
        with open(Path(config["out_folder"], "translit_train.pickle"), "wb") as outf:
            pickle.dump(train_dataset, outf)

    # Construct eval data, discarded 38 sentences
    if Path(config["out_folder"], "translit_dev.pickle").is_file():
        with open(Path(config["out_folder"], "translit_dev.pickle"), "rb") as f:
            eval_dataset = pickle.load(f)
    else:
        eval_dat = {}
        with open(config["eval_path"], encoding="utf-8") as eval_json:
            eval_dat = json.load(eval_json)
        eval_dataset = construct_dataset(eval_dat, translit, config["eval_graphml"])
        with open(Path(config["out_folder"], "translit_dev.pickle"), "wb") as outf:
            pickle.dump(eval_dataset, outf)

    # Display an example datapoint
    pp.pprint(train_dataset[-1])
    pp.pprint(eval_dataset[-1])

    # Index data
    logger.info("Index train data")
    train_data_indexed = index_dataset(train_dataset, char2index)
    eval_data_indexed = index_dataset(eval_dataset, char2index)

    # Display an example
    pp.pprint(eval_dataset[-1])
    pp.pprint(eval_data_indexed[-1])
    assert len(eval_dataset[-1]["sandhied_merged"]) == eval_data_indexed[-1][0].size(
        dim=0
    )

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

    # Build model
    logger.info("Build model")
    model = build_model(config, vocabulary)
    use_cuda = config["cuda"]
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using devide: {device}")

    model = model.to(device)

    # Build optimizer
    optimizer = build_optimizer(model, config)

    # Train
    logger.info("Train\n")
    epochs = config["epochs"]
    criterion = get_loss(config)
    start = time.time()

    model, optimizer = train(
        model, criterion, optimizer, train_dataloader, epochs, device
    )

    # Save model
    name = config["name"]
    save_model(model, optimizer, vocabulary, char2index, index2char, name)

    # Predictions
    predictions, true_unsandhied, split_predictions = make_predictions(
        model, eval_dataloader, eval_dataset, device, translit
    )

    # (false) end of prediction
    duration = time.time() - start

    # Example prediction
    idx = 10

    logger.info(split_predictions[idx])
    logger.info(
        [
            eval_dataset[idx]["sandhied"][i:j]
            for i, j in zip(
                [0] + split_predictions[idx], split_predictions[idx] + [None]
            )
        ]
    )
    logger.info(eval_dataset[idx]["sandhied"])
    logger.info(predictions[idx])
    logger.info(true_unsandhied[idx])

    # Create submission
    save_task1_predictions(predictions, duration)

    # Evaluation
    scores = evaluate(true_unsandhied, predictions, task_id="t1")
