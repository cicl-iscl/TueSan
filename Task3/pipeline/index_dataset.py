"""
Make vocabulary.
This means we collect all distinct characters that appear in the dataset
in the input sequence only.
"""

import torch

from functools import partial
from sandhi_rules import KEEP_RULE
from stemming_rules import UNK_RULE
from torch.nn.utils.rnn import pad_sequence

pad = partial(pad_sequence, batch_first=True)

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
    source, sandhi_target, stem_target, tag_target, boundaries = zip(*batch)
    source = pad(source)
    sandhi_target = pad(sandhi_target)

    num_tokens = sum([len(sent) for sent in boundaries])
    stem_target = torch.cat(stem_target, dim=0).flatten().long()
    tag_target = torch.cat(tag_target, dim=0).flatten().long()

    try:
        assert num_tokens == len(stem_target) == len(tag_target)
    except AssertionError:
        print(stem_target)
        print(tag_target)
        print(boundaries)

    return (source, sandhi_target, stem_target, tag_target, boundaries)


def eval_collate_fn(batch):
    raw_source, source = zip(*batch)
    return raw_source, pad(source)


class Indexer:
    def __init__(self, vocabulary, sandhi_rules, stem_rules, tags):
        self.vocabulary = vocabulary

        self.char2index = {char: index for index, char in enumerate(self.vocabulary)}
        self.index2char = {index: char for char, index in self.char2index.items()}
        self.unk_index = self.char2index[UNK_TOKEN]

        self.sandhi_rules = sandhi_rules
        self.sandhi_rule2index = {
            rule: index + 1 for index, rule in enumerate(self.sandhi_rules)
        }
        self.index2sandhi_rule = {
            index: rule for rule, index in self.sandhi_rule2index.items()
        }
        self.keep_index = self.sandhi_rule2index[KEEP_RULE]

        self.stem_rules = stem_rules
        self.stem_rule2index = {
            rule: index + 1 for index, rule in enumerate(self.stem_rules)
        }
        self.stem_rule2index[UNK_RULE] = 0
        self.index2stem_rule = {
            index: rule for rule, index in self.stem_rule2index.items()
        }
        self.unk_stem_rule_index = self.stem_rule2index[UNK_RULE]

        self.tags = [UNK_TOKEN] + list(sorted(set(tags)))
        self.tag2index = {tag: index for index, tag in enumerate(self.tags)}
        self.index2tag = {index: tag for tag, index in self.tag2index.items()}
        self.unk_tag_index = self.tag2index[UNK_TOKEN]

    def index_sent(self, sent):
        index = lambda char: self.char2index.get(char, self.unk_index)
        indexed_sent = list(map(index, sent))
        indexed_sent = torch.LongTensor(indexed_sent)

        return indexed_sent

    def restore_indexed_sent(self, indexed_sent):
        restore = lambda idx: self.index2char.get(idx, UNK_TOKEN)
        restored_sent = "".join(map(restore, indexed_sent))

        return restored_sent

    def index_sandhi_rules(self, rules):
        assert isinstance(rules, list)
        index = lambda rule: self.sandhi_rule2index.get(rule, self.keep_index)
        indexed_rules = list(map(index, rules))
        indexed_rules = torch.LongTensor(indexed_rules)

        return indexed_rules

    def restore_indexed_sandhi_rules(self, indexed_rules):
        restore = lambda idx: self.index2sandhi_rule.get(idx, KEEP_RULE)
        restored_rules = list(map(restore, indexed_rules))

        return restored_rules

    def index_stem_rules(self, rules):
        assert isinstance(rules, list)
        index = lambda rule: self.stem_rule2index.get(rule, self.unk_stem_rule_index)
        indexed_rules = list(map(index, rules))
        indexed_rules = torch.LongTensor(indexed_rules)

        return indexed_rules

    def restore_indexed_stem_rules(self, indexed_rules):
        restore = lambda idx: self.index2stem_rule.get(idx, UNK_RULE)
        restored_rules = list(map(restore, indexed_rules))

        return restored_rules

    def index_tags(self, tags):
        assert isinstance(tags, list)
        index = lambda tag: self.tag2index.get(tag, self.unk_tag_index)
        indexed_tags = list(map(index, tags))
        indexed_tags = torch.LongTensor(indexed_tags)

        return indexed_tags

    def restore_indexed_tags(self, indexed_tags):
        restore = lambda idx: self.index2tag.get(idx, UNK_TOKEN)
        restored_tags = list(map(restore, indexed_tags))

        return restored_tags


def make_indexer(data, sandhi_rules, stem_rules, tags):
    vocabulary = set()

    for joint_sent, *_ in data:
        # whitespaces are kept
        # Just add all characters in input sentence to vocab
        vocabulary.update(set(joint_sent))

    # Add special tokens to vocabulary
    vocabulary = specials + list(sorted(vocabulary))

    # Make indexer: indices <-> internal characters <-> unicode characters
    indexer = Indexer(vocabulary, sandhi_rules, stem_rules, tags)

    return indexer


def index_dataset(train_data, eval_data, sandhi_rules, stem_rules, tags):
    indexer = make_indexer(train_data, sandhi_rules, stem_rules, tags)

    # Index train data
    indexed_train_data = []
    for source, sandhi_target, stem_target, tag_target, boundaries in train_data:
        indexed_source = indexer.index_sent(source)
        indexed_sandhi_target = indexer.index_sandhi_rules(sandhi_target)
        indexed_stem_target = indexer.index_stem_rules(stem_target)
        indexed_tag_target = indexer.index_tags(tag_target)

        indexed_train_data.append(
            (
                indexed_source,
                indexed_sandhi_target,
                indexed_stem_target,
                indexed_tag_target,
                boundaries,
            )
        )

    # Index eval data
    indexed_eval_data = []
    for raw_source, *_ in eval_data:
        indexed_source = indexer.index_sent(raw_source)
        indexed_eval_data.append((raw_source, indexed_source))

    return indexed_train_data, indexed_eval_data, indexer
