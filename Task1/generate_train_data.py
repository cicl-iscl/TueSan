"""Generate training data for Task 1
	with or without extra translit.

	For alignment we need to also convert the tokens.
"""

import numpy as np
import networkx as nx
import edit_distance
from edit_distance import SequenceMatcher as align

from tqdm import tqdm
from pathlib import Path
from uni2intern import internal_transliteration_to_unicode as to_uni
from uni2intern import unicode_to_internal_transliteration as to_intern
from helpers import get_task1_IO


def get_data(datapoint, translit=False):
    sandhied, unsandhied_tokenized = get_task1_IO(datapoint, translit=translit)
    unsandhied = " ".join(unsandhied_tokenized)
    if translit:
        unsandhied = to_intern(unsandhied)
    # Punctuation are not removed from input

    return sandhied, unsandhied, unsandhied_tokenized


def tokenize(sandhied, unsandhied):
    # Align unsandhied tokens with sandhied sentence
    # to get split points
    alignment = align(a=sandhied, b=unsandhied)
    alignment = alignment.get_matching_blocks()
    sandhied_indices, unsandhied_indices, _ = zip(*alignment)

    # Remove dummy indices
    sandhied_indices = sandhied_indices[:-1]
    unsandhied_indices = unsandhied_indices[:-1]

    # Find indices with spaces -> split
    split_indices = [index for index, char in enumerate(sandhied) if not char.strip()]

    # Find indices where sandhied/unsandhied sentence is
    # not aligned -> split here
    sandhied_pointer = 0
    unsandhied_pointer = 0

    for sandhied_index, unsandhied_index in zip(sandhied_indices, unsandhied_indices):
        if sandhied_pointer != sandhied_index or unsandhied_pointer != unsandhied_index:
            if (
                sandhied_index + 1 not in split_indices
                and sandhied_index - 1 not in split_indices
            ):
                split_indices.append(sandhied_index)
                sandhied_pointer = sandhied_index
                unsandhied_pointer = unsandhied_index

        sandhied_pointer += 1
        unsandhied_pointer += 1

    split_indices = [0] + list(sorted(set(split_indices)))

    # Tokenize sandhied sentence:
    # Split at split indices
    split_indices = zip(split_indices, split_indices[1:] + [None])
    tokens = [sandhied[start:stop] for start, stop in split_indices]
    tokens = [token.strip() for token in tokens if not token.isspace()]

    return tokens


def remove_trailing_syllables(tokens, unsandhied_tokenized, translit=False):
    if translit:
        if (
            len(tokens) > len(unsandhied_tokenized)
            and len(tokens) > 2
            and to_intern(unsandhied_tokenized[-1]).endswith(tokens[-1])
            and [to_intern(t) for t in unsandhied_tokenized[:3]]
            == tokens[-2].strip()[:3]
        ):

            tokens = tokens[:-1]

    elif (
        len(tokens) > len(unsandhied_tokenized)
        and len(tokens) > 2
        and unsandhied_tokenized[-1].endswith(tokens[-1])
        and unsandhied_tokenized[:3] == tokens[-2].strip()[:3]
    ):

        tokens = tokens[:-1]

    return tokens


def merge_single_letters(tokens):
    merged_tokens = []
    skip_next = False

    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue

        token = token.strip()
        if len(token.strip()) == 1 and index < len(tokens) - 1:
            merged_token = token + tokens[index + 1].strip()
            merged_tokens.append(merged_token)
            skip_next = True
        else:
            merged_tokens.append(token)

    return merged_tokens


def make_labels(tokens):
    combined_string = "_".join(tokens)  # keep whitespaces as underscores

    offsets = np.array([len(token) for token in tokens])
    split_indices = np.cumsum(offsets) - 1

    labels = np.zeros(len(combined_string), dtype=np.int8)
    labels[split_indices] = 1

    return combined_string, labels


def get_allowed_words(sent_id, graphml_folder=None, translit=False):
    graphml_path = Path(graphml_folder, str(str(sent_id) + ".graphml"))
    with open(graphml_path, mode="rb") as graph_input:
        graph = nx.read_graphml(graph_input)
    if translit:
        allowed_words = [to_intern(node[1]["word"]) for node in graph.nodes(data=True)]
    else:
        allowed_words = [node[1]["word"] for node in graph.nodes(data=True)]

    return list(sorted(set(allowed_words)))


def construct_dataset(data, translit=False, graphml_folder=None):
    dataset = []
    cnt = 0

    for sentence in tqdm(data):
        sent_id = sentence["sent_id"]
        sandhied, unsandhied, unsandhied_tokenized = get_data(sentence, translit)

        # Filter sents where sandhied >> unsandhied
        # These are most likely wrong
        if len(sandhied) / len(unsandhied) > 1.5:
            cnt += 1
            continue

        tokens = tokenize(sandhied, unsandhied)
        tokens = remove_trailing_syllables(tokens, unsandhied_tokenized, translit)
        tokens = merge_single_letters(tokens)

        sandhied_merged, labels = make_labels(tokens)
        allowed_words = get_allowed_words(sent_id, graphml_folder, translit)

        datapoint = {
            "sandhied_merged": sandhied_merged,
            "labels": labels,
            "tokens": tokens,
            "allowed_words": allowed_words,
            "unsandhied": unsandhied.split(),
            "sent_id": sent_id,
        }

        dataset.append(datapoint)
        # break

    print(f"{cnt} sentences discarded during dataset construction.")

    return dataset
