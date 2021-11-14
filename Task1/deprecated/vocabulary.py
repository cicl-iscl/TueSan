"""
Make vocabulary.
This means we collect all distinct characters that appear in the dataset
in the input sequence only.
"""

import torch

from generate_dataset import KEEP_RULE

# We need a couple of special tokens:
# Decoding: Need start of sequence (<S>) and end of sequence (</S>) tokens
#           s.t. we know where to stop
# Inference: Need <UNK> token if we encounter unknown character
# Padding: Need <PAD> token
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<S>"
EOS_TOKEN = "</S>"
specials = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


def train_collate_fn(batch):
    source, target = zip(*batch)
    source = pad_sequence(source)
    target = pad_sequence(target)

    return source, target


def eval_collate_fn(batch):
    return pad_sequence(batch)


class Indexer:
    def __init__(self, vocabulary, rules):
        self.vocabulary = vocabulary

        self.char2index = {char: index for index, char in enumerate(self.vocabulary)}
        self.index2char = {index: char for char, index in self.char2index.items()}
        self.unk_index = self.char2index[UNK_TOKEN]

        self.rules = rules
        self.rule2index = {rule: index for index, rule in enumerate(self.rules)}
        self.index2rule = {index: rule for rule, index in self.rule2index}
        self.keep_index = self.rule2index[KEEP_RULE]

    def index_sent(self, sent):
        assert isinstance(sent, str)
        index = lambda char: self.char2index.get(char, self.unk_index)
        indexed_sent = list(map(index, sent))
        indexed_sent = torch.LongTensor(indexed_sent)

        return indexed_sent

    def restore_indexed_sent(self, indexed_sent):
        restore = lambda idx: self.index2char.get(idx, UNK_TOKEN)
        restored_sent = "".join(map(restore, indexed_sent))

        return restored_sent

    def index_rules(self, rules):
        assert isinstance(rules, list)
        index = lambda rule: self.rule2index.get(rule, self.keep_index)
        indexed_rules = list(map(index, rules))
        indexed_rules = torch.LongTensor(indexed_rules)

        return indexed_rules

    def restore_indexed_rules(indexed_rules):
        restore = lambda idx: self.index2rule.get(idx, KEEP_RULE)
        restored_sent = list(map(restore, indexed_rules))

        return restored_sent


def make_indexer(data, rules):
    vocabulary = set()

    for joint_sent, _ in data:
        # whitespaces are kept
        # Just add all characters in input sentence to vocab
        vocabulary.update(set(joint_sent))

    # Add special tokens to vocabulary
    vocabulary = specials + list(sorted(vocabulary))

    # Make indexer: indices <-> internal characters <-> unicode characters
    indexer = Indexer(vocabulary, rules)

    return indexer


def index_data(train_data, eval_data, rules):
    indexer = make_indexer(train_data, rules)

    # Index train data
    indexed_train_data = []
    for source, target in train_data:
        indexed_source = indexer.index_sent(source)
        indexed_target = indexer.index_rules(target)

        indexed_train_data.append((indexed_source, indexed_target))

    # Index eval data
    indexed_eval_data = list(map(indexer.index_sent, eval_data))

    return indexed_train_data, indexed_eval_data, indexer
