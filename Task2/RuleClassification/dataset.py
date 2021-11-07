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
from extract_rules import UNK_RULE


def index_tokens(tokens, char2index):
    indexed_input = [[char2index[char] for char in token] for token in tokens]
    indexed_input = [torch.LongTensor(token) for token in indexed_input]
    return indexed_input


def index_dataset(data, char2index, rule_encoder, tag_encoder=None, eval=False):
    """
    Dataset indexing:
    This means, we need to convert the strings (inputs or stems) to the
    vocabulary indices. We already store the indices as PyTorch tensors.
    Furthermore, we convert labels to integers via the encoder from the
    previous step.
    """
    indexed_dataset = []

    for sentence in tqdm(data):
        indexed_tokens = []
        indexed_rules = []
        indexed_tags = []

        tokens, stems, tags, rules = zip(*sentence)
        # Index tokens
        indexed_tokens = index_tokens(tokens, char2index)
        # Index rules
        indexed_rules = [rule_encoder[rule] for rule in rules]
        indexed_rules = torch.LongTensor(indexed_rules)
        # Index tags
        if tag_encoder is not None:
            indexed_tags = [tag_encoder[tag] for tag in tags]
            indexed_tags = torch.LongTensor(indexed_tags)

        # Save data
        sentence = (indexed_tokens, indexed_rules)
        if tag_encoder is not None:
            sentence = (*sentence, indexed_tags)

        if eval:
            sentence = (tokens, stems, *sentence)

        indexed_dataset.append(sentence)

    return indexed_dataset


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
    max_word_length = max([max([len(word) for word in sent]) for sent in inputs])

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


def collate_fn(batch):
    """
    To create a minibatch, we simply pad all data accordingly
    """
    if len(batch[0]) == 2:
        tokens, rules = zip(*batch)
        tokens = pad2d(tokens)
        rules = pad_sequence(rules, padding_value=0, batch_first=True)

        return tokens, rules

    elif len(batch[0]) == 3:
        tokens, rules, tags = zip(*batch)
        tokens = pad2d(tokens)
        rules = pad_sequence(rules, padding_value=0, batch_first=True)
        tags = pad_sequence(tags, padding_value=0, batch_first=True)

        return tokens, rules, tags


def eval_collate_fn(batch):
    """
    We need a separate batch collation function for evaluation,
    because for evaluation we need the raw (not indexed) input
    strings.
    """
    if len(batch[0]) == 4:
        raw_tokens, stems, tokens, rules = zip(*batch)
        tokens = pad2d(tokens)
        rules = pad_sequence(rules, padding_value=0, batch_first=True)

        return raw_tokens, stems, tokens, rules

    elif len(batch[0]) == 5:
        raw_tokens, stems, tokens, rules, tags = zip(*batch)
        tokens = pad2d(tokens)
        rules = pad_sequence(rules, padding_value=0, batch_first=True)
        tags = pad_sequence(tags, padding_value=0, batch_first=True)

        return raw_tokens, stems, tokens, rules, tags
