import torch
import json

import time
import pprint

pp = pprint.PrettyPrinter(indent=4)

from functools import partial
from torch.utils.data import DataLoader

from helpers import load_data, save_task2_predictions
from vocabulary import make_vocabulary, PAD_TOKEN
from dataset import index_dataset, collate_fn, eval_collate_fn
from model import build_model, build_optimizer, save_model
from training import train
from evaluate import evaluate_model, print_metrics, format_predictions
from extract_rules import get_token_rule_mapping, get_rules
from scoring import evaluate


# Ignore warning (who cares?)
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    with open("config.cfg") as config:
        config = json.load(config)

    translit = config["translit"]

    # Load data
    print("\nLoad data")
    train_data = load_data(config["train_path"], translit)
    eval_data = load_data(config["eval_path"], translit)
    
    train_data = train_data[:128]

    # Exctract rules
    print("\nExtracting rules")
    use_tag = config["rules_use_tag"]
    min_rule_frequency = config["rules_min_frequency"]
    rules = get_rules(train_data, use_tag=use_tag, translit=translit)
    rules = [rule for rule, count in rules.items() if count > min_rule_frequency]
    print(f"Extracted {len(rules)} rules.")

    # Convert train dataset
    print("\nConverting train dataset")
    train_dataset = get_token_rule_mapping(
        train_data, rules, use_tag=use_tag, translit=translit
    )

    # Convert eval dataset
    print("\nConverting eval dataset")
    eval_dataset = get_token_rule_mapping(
        eval_data, rules, use_tag=use_tag, translit=translit
    )

    # Make vocabulary
    print("Make vocab")
    if not use_tag:
        vocabulary, rule_encoder, tag_encoder, char2index, index2char = make_vocabulary(
            train_dataset, use_tag=False
        )
    else:
        vocabulary, rule_encoder, char2index, index2char = make_vocabulary(
            train_dataset, use_tag=True
        )
        tag_encoder = None

    # Index data
    print("\nIndex train data")
    train_data_indexed = index_dataset(
        train_dataset, char2index, rule_encoder, tag_encoder=tag_encoder
    )
    print("\nIndex eval data")
    eval_data_indexed = index_dataset(
        eval_dataset, char2index, rule_encoder, tag_encoder=tag_encoder, eval=True
    )

    # Build dataloaders
    batch_size = config["batch_size"]
    train_dataloader = DataLoader(
        train_data_indexed[:128],
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        eval_data_indexed,
        batch_size=batch_size,
        collate_fn=eval_collate_fn,
        shuffle=False,
    )

    # Build model
    print("\nBuild model")
    if use_tag:
        num_classes = [len(rule_encoder) + 2]
        names = ["rules"]
    else:
        num_classes = [len(rule_encoder) + 2, len(tag_encoder) + 2]
        names = ["rules", "tags"]

    model = build_model(config, vocabulary, num_classes, names)
    use_cuda = config["cuda"]
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using devide: {device}")

    model = model.to(device)

    # Build optimizer
    print("\nBuild optimizer")
    optimizer = build_optimizer(model, config)

    # Train
    print("\nTrain")
    epochs = config["epochs"]
    print(f"Training for {epochs} epochs\n")

    start = time.time()

    model, optimizer = train(model, optimizer, train_dataloader, epochs, device)

    # Save model
    print("\nSaving model")
    name = config["name"]
    save_model(
        model,
        optimizer,
        vocabulary,
        rule_encoder,
        tag_encoder,
        char2index,
        index2char,
        name,
    )

    # Evaluate
    print("\nPredicting")
    eval_predictions = evaluate_model(
        model, eval_dataloader, device, rules, rule_encoder, tag_encoder
    )
    formatted_predictions = format_predictions(eval_predictions)
    pp.pprint(formatted_predictions[:2])

    duration = time.time() - start
    print(f"Duration: {duration:.2f} seconds.\n")

    # Create submission
    print("Create submission files")
    save_task2_predictions(formatted_predictions, duration)

    # Evaluate
    print("\nEvaluating")
    print_metrics(eval_predictions, eval_dataset)
    # Task 2 Evaluation
    scores = evaluate(eval_data[1], formatted_predictions, task_id="t2")

# pp.pprint(formatted_predictions[0])
# print(type(formatted_predictions))
# print(type(formatted_predictions[0]))
# pp.pprint(eval_data[0])
# print(type(eval_data))
# print(type(eval_data[0]))
