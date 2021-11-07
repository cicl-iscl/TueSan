import torch
import json

from functools import partial
from torch.utils.data import DataLoader

from helpers import load_data, load_sankrit_dictionary
from vocabulary import make_vocabulary, PAD_TOKEN
from dataset import index_dataset, collate_fn, eval_collate_fn
from model import build_model, build_optimizer, save_model
from training import train
from evaluate import evaluate_model, print_metrics

# Ignore warning (who cares?)
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    with open("config.cfg") as config:
        config = json.load(config)

    # Load data
    print("Load data")
    train_data = load_data(config["train_path"])
    eval_data = load_data(config["eval_path"])

    # Make vocabulary
    print("Make vocab")
    # TODO: Include UNK chars and labels
    vocabulary, tag_encoder, char2index, index2char = make_vocabulary(train_data)
    pad_tag = tag_encoder.transform([PAD_TOKEN]).item()

    # Index data
    print("Index train data")
    train_data_indexed, train_data, discarded = index_dataset(
        train_data, char2index, tag_encoder
    )
    print(f"Discarded {discarded} sentences from train data")
    print("Index eval data")
    eval_data_indexed, eval_data, discarded = index_dataset(
        eval_data, char2index, tag_encoder, eval=True
    )
    print(f"Discarded {discarded} sentences from eval data")

    # Build dataloaders
    batch_size = config["batch_size"]
    collate_fn = partial(collate_fn, pad_tag=pad_tag)
    eval_collate_fn = partial(eval_collate_fn, pad_tag=pad_tag)

    train_dataloader = DataLoader(
        train_data_indexed, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_data_indexed,
        batch_size=batch_size,
        collate_fn=eval_collate_fn,
        shuffle=False,
    )

    # Build model
    print("Build model")
    model = build_model(config, vocabulary, tag_encoder)
    use_cuda = config["cuda"]
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using devide: {device}")

    model = model.to(device)

    # Build optimizer
    optimizer = build_optimizer(model, config)

    # Train
    print("Train\n")
    epochs = config["epochs"]
    model, optimizer = train(
        model, optimizer, train_dataloader, epochs, device, pad_tag
    )

    # Save model
    name = config["name"]
    save_model(model, optimizer, vocabulary, tag_encoder, char2index, index2char, name)

    # Evaluate
    mode = config["eval_mode"]
    dictionary_path = config.get("dictionary_path", None)
    if dictionary_path is None:
        dictionary = None
    else:
        dictionary = load_sankrit_dictionary(dictionary_path)

    eval_predictions = evaluate_model(
        model,
        eval_dataloader,
        tag_encoder,
        char2index,
        index2char,
        device,
        mode=mode,
        dictionary=dictionary,
    )

    print_metrics(eval_predictions, eval_data)
