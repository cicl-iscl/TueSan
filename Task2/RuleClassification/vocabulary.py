"""
Make vocabulary and encode labels.
This means we collect all distinct characters that appear in the dataset
either in the input sentence or in the target stems.
Likewise, we collect all morphological tags.
"""

import numpy as np
from collections import defaultdict

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

from extract_rules import UNK_RULE


def make_vocabulary(data, use_tag=True):
    tags = set()
    vocabulary = set()
    rules = set()

    for sentence in data:
        for token, stem, tag, rule in sentence:
            # Just add all characters in input sentence to vocab
            vocabulary.update(set(token))
            rules.add(rule)

            if not use_tag:
                tags.add(tag)

    # Add special tokens to vocabulary
    vocabulary = specials + list(sorted(vocabulary))
    # We also need a padding tag
    tags = list(sorted(tags))

    # Same for rules:
    rules = list(rules)

    # Only make tag encoder if we use tags
    if not use_tag:
        # Convert tags (=str) to integers
        tag_encoder = defaultdict(lambda: 1)
        tag_encoder.update({tag: i + 2 for i, tag in enumerate(tags)})

    # Convert rules to integers
    rule_encoder = defaultdict(lambda: 1)
    rule_encoder.update({rule: i + 2 for i, rule in enumerate(rules)})
    print(f"We are working with {len(rule_encoder)} rules")

    # Make dictionary: characters <-> indices
    char2index = {char: index for index, char in enumerate(vocabulary)}
    index2char = {index: char for char, index in char2index.items()}

    if not use_tag:
        return vocabulary, rule_encoder, tag_encoder, char2index, index2char
    else:
        return vocabulary, rule_encoder, char2index, index2char
