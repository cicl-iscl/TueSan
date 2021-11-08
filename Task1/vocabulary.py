"""
Make vocabulary.
This means we collect all distinct characters that appear in the dataset
in the input sequence only.
"""
from uni2intern import internal_transliteration_to_unicode as to_uni

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
    vocabulary = set()

    for joint_sent, _ in data:
        # whitespaces to underscores
        joint_sent = joint_sent.replace(" ", "_")
        # Just add all characters in input sentence to vocab
        vocabulary.update(set(joint_sent))

    # Add special tokens to vocabulary
    vocabulary = specials + list(sorted(vocabulary))

    # Make dictionary: indices <-> internal characters <-> unicode characters
    char2index = {char: index for index, char in enumerate(vocabulary)}
    index2char = {index: char for char, index in char2index.items()}
    char2uni = {char: to_uni(char) for char in vocabulary}
    char2uni["_"] = " "

    return vocabulary, char2index, index2char, char2uni
