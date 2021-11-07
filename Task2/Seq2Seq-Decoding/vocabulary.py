"""
Make vocabulary and encode labels.
This means we collect all distinct characters that appear in the dataset
either in the input sentence or in the target stems.
Likewise, we collect all morphological tags.
"""

# Use sklearn's LabelEncoder for simplicity:
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn.preprocessing import LabelEncoder

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


def make_vocabulary(data):
    tags = set()
    vocabulary = set()

    for input, output in data:
        # Just add all characters in input sentence to vocab
        vocabulary.update(set(input))

        for stem, tag in output:
            # Add all characters in stem to vocab
            vocabulary.update(set(stem))
            # Add tag to tags
            tags.add(tag)

    # Add special tokens to vocabulary
    vocabulary = specials + list(sorted(vocabulary))
    # We also need a padding tag
    tags = list(sorted(tags)) + [PAD_TOKEN, UNK_TOKEN]

    # Convert tags (=str) to integers
    tag_encoder = LabelEncoder()
    tag_encoder.fit(tags)
    pad_tag = tag_encoder.transform([PAD_TOKEN]).item()

    # Make dictionary: characters <-> indices
    char2index = {char: index for index, char in enumerate(vocabulary)}
    index2char = {index: char for char, index in char2index.items()}

    return vocabulary, tag_encoder, char2index, index2char
