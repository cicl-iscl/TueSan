"""
Implements functionality for making the data
processable by PyTorch.
"""

import torch
import numpy as np

from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from vocabulary import SOS_TOKEN, EOS_TOKEN, UNK_TOKEN


def validate_tag(tag, tag_encoder):
    if tag in tag_encoder.classes_:
        return tag
    else:
        return UNK_TOKEN


def index_dataset(data, char2index, tag_encoder, eval=False):
    """
    Dataset indexing:
    This means, we need to convert the strings (inputs or stems) to the
    vocabulary indices. We already store the indices as PyTorch tensors.
    Furthermore, we convert labels to integers via the encoder from the
    previous step.
    """
    indexed_dataset = []
    data_updated = []
    discarded = 0

    for input, output in tqdm(data):
        # Whitespace tokenize sentence
        input = input.split()
        stems, tags = zip(*output)

        # Only save sentences if number of input tokens
        # is the same as number of ground truth stems/labels
        if not eval:
            if len(input) != len(tags) or len(input) != len(stems):
                discarded += 1
                continue

        # Index input sentence
        # Discard sentences with unknown characters
        try:
            indexed_input = [
                [char2index[char] for char in token] for token in input
            ]
            indexed_input = [
                torch.LongTensor(token) for token in indexed_input
            ]
        except KeyError:
            discarded += 1
            continue

        # Index stems
        try:
            # Prepend start of sequence token and append end of sequence token to stems
            stems = [[SOS_TOKEN] + list(stem) + [EOS_TOKEN] for stem in stems]
            indexed_stems = [
                [char2index[char] for char in stem] for stem in stems
            ]
            indexed_stems = [torch.LongTensor(stem) for stem in indexed_stems]
        except KeyError:
            discarded += 1
            continue

        # Index tags
        # If tag is unknown, replace with UNK_TOKEN
        tag_validator = partial(validate_tag, tag_encoder=tag_encoder)
        tags = list(map(tag_validator, tags))
        indexed_tags = torch.from_numpy(tag_encoder.transform(tags)).float()

        # Save data
        data_updated.append((input, output))
        if eval:
            dp = (input, indexed_input, indexed_stems, indexed_tags)
            indexed_dataset.append(dp)
        else:
            dp = (indexed_input, indexed_stems, indexed_tags)
            indexed_dataset.append(dp)

    return indexed_dataset, data_updated, discarded


# Now, we implement padding for each minibatch.
# Because dimensions in the minibatch tensors must match for all samples,
# but e.g. input sentences have different lengths, we add padding tokens/labels.
#
# For input sentences and stems, we have to pad in 2 dimensions:
# First, sentences may have different numbers of words.
# Secondly, words may have different number of characters.
#
# We need to represent both dimensions, because tag classification is on
# word level, but requires access to sub-word info (e.g. endings).
# Therefore, our model is character-based.


def pad2d(inputs):
    """Pads input in token and character dimensions"""
    padded_batch = []

    # Get maximum number of tokens in sentence
    max_sent_length = max([len(sent) for sent in inputs])
    # Get maximum number of characters in token
    max_word_length = max(
        [max([len(word) for word in sent]) for sent in inputs]
    )

    for sent in inputs:
        # We process each sentence individually
        # First, we pad the sentence so that all indexed tokens
        # are padded to same length (character dimension)
        sent = pad_sequence(sent, batch_first=True)
        assert len(sent.shape) == 2

        # Then, we concatenate padding to the character dimension
        # to reach the length of the longest word in the minibatch
        # The longest word can be in a different sentence
        #
        # Max. word length of current sentence
        word_length = sent.shape[1]
        # Difference to max. word length in minibatch
        char_padding_length = max_word_length - word_length
        # Create padding (a lot of 0s)
        char_padding = sent.new_zeros(sent.shape[0], char_padding_length)
        sent = torch.cat([sent, char_padding], dim=1)

        # Finally, we concatenate padding to the word dimension
        # to reach the length of the longest sentence in the minibatch
        #
        # Length of current sentence (in words)
        sent_length = sent.shape[0]
        # Difference to longest sentence
        token_padding_length = max_sent_length - sent_length
        # Concatenate padding
        token_padding = sent.new_zeros(token_padding_length, max_word_length)
        sent = torch.cat([sent, token_padding], dim=0)

        padded_batch.append(sent)

    return torch.stack(padded_batch)


def collate_fn(batch, pad_tag):
    """
    To create a minibatch, we simply pad all data accordingly
    """
    inputs, stems, labels = zip(*batch)

    inputs = pad2d(inputs)
    stems = pad2d(stems)
    labels = pad_sequence(labels, padding_value=pad_tag, batch_first=True)

    return inputs, stems, labels


def eval_collate_fn(batch, pad_tag):
    """
    We need a separate batch collation function for evaluation,
    because for evaluation we need the raw (not indexed) input
    strings.
    """
    raw_inputs, inputs, stems, labels = zip(*batch)

    inputs = pad2d(inputs)
    stems = pad2d(stems)
    labels = pad_sequence(labels, padding_value=pad_tag, batch_first=True)

    return raw_inputs, inputs, stems, labels
