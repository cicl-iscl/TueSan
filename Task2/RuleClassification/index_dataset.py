"""
Make vocabulary.
This means we collect all distinct characters that appear in the dataset
in the input sequence only.
"""

import torch

from tqdm import tqdm
from stemming_rules import UNK_RULE
from torch.nn.utils.rnn import pad_sequence

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


def train_collate_fn(batch):
    """
    To create a minibatch, we simply pad all data accordingly
    """
    if len(batch[0]) == 2:
        tokens, rules = zip(*batch)
        tokens = pad2d(tokens)
        rules = torch.cat(rules, dim=0).flatten().long()

        return tokens, rules

    elif len(batch[0]) == 3:
        tokens, rules, tags = zip(*batch)
        tokens = pad2d(tokens)
        rules = torch.cat(rules, dim=0).flatten().long()
        tags = torch.cat(tags, dim=0).flatten().long()

        return tokens, rules, tags


def eval_collate_fn(batch):
    raw_tokens, indexed_tokens = zip(*batch)
    return raw_tokens, pad2d(indexed_tokens)


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


class Indexer:
    def __init__(self, vocabulary, stem_rules, tags):
        self.vocabulary = vocabulary

        self.char2index = {char: index for index, char in enumerate(self.vocabulary)}
        self.index2char = {index: char for char, index in self.char2index.items()}
        self.unk_index = self.char2index[UNK_TOKEN]

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

    def index_token(self, token):
        index = lambda char: self.char2index.get(char, self.unk_index)
        indexed_token = list(map(index, token))
        indexed_token = torch.LongTensor(indexed_token)

        return indexed_token

    # def restore_indexed_sent(self, indexed_sent):
    #    restore = lambda idx: self.index2char.get(idx, UNK_TOKEN)
    #    restored_sent = "".join(map(restore, indexed_sent))

    #    return restored_sent

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


def make_indexer(data, stem_rules, tags):
    vocabulary = set()

    for tokens, *_ in data:
        # whitespaces are kept
        # Just add all characters in input sentence to vocab
        joint_sent = " ".join(tokens)
        vocabulary.update(set(joint_sent))

    # Add special tokens to vocabulary
    vocabulary = specials + list(sorted(vocabulary))

    # Make indexer: indices <-> internal characters <-> unicode characters
    indexer = Indexer(vocabulary, stem_rules, tags)

    return indexer


def index_dataset(train_data, eval_data, stem_rules, tags, tag_rules):
    indexer = make_indexer(train_data, stem_rules, tags)

    # Index train data
    indexed_train_data = []
    for tokens, stem_target, tag_target in tqdm(train_data):
        indexed_tokens = list(map(indexer.index_token, tokens))
        indexed_stem_target = indexer.index_stem_rules(stem_target)
        indexed_tag_target = indexer.index_tags(tag_target)

        if tag_rules:
            indexed_train_data.append((indexed_tokens, indexed_stem_target))
        else:
            indexed_train_data.append(
                (indexed_tokens, indexed_stem_target, indexed_tag_target)
            )

    # Index eval data
    indexed_eval_data = []
    for raw_tokens, *_ in eval_data:
        raw_tokens = raw_tokens.split()
        indexed_tokens = list(map(indexer.index_token, raw_tokens))
        indexed_eval_data.append((raw_tokens, indexed_tokens))

    return indexed_train_data, indexed_eval_data, indexer
