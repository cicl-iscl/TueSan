"""
Implements functionality for making the data
processable by PyTorch.
"""

import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader


def index_dataset(data, char2index):
    """From preconstructed dataset...
    """
    indexed_dataset = []
    data_updated = []
    discarded = 0

    for dp in tqdm(data):
        # Whitespaces are already replaced with underscores in 'sandhied_merged'
        text, labels = dp['sandhied_merged'], dp['labels']
        indexed_text = [char2index[char] for char in text]
        indexed_text = torch.LongTensor(indexed_text)
        labels = torch.from_numpy(labels).float()
        indexed_dataset.append([indexed_text, labels])

    return indexed_dataset


def pad(batch):
    inputs, labels = zip(*batch)
    max_len = max([len(seq) for seq in inputs])

    padded_batch = []
    for seq, labels in batch:
        padding_length = max_len - len(seq)
        seq_padding = seq.new_zeros(padding_length)
        padded_seq = torch.cat([seq, seq_padding], dim=-1)
        label_padding = labels.new_zeros(padding_length)
        padded_labels = torch.cat([labels, label_padding], dim=-1)

        padded_batch.append((padded_seq, padded_labels))

    return padded_batch


def collate_fn(batch):
    padded_batch = pad(batch)
    padded_inputs, padded_labels = zip(*padded_batch)

    inputs = torch.stack(padded_inputs)
    labels = torch.stack(padded_labels)

    return inputs, labels
