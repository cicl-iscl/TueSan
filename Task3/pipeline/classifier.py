"""
Implementation of encoder for Sanskrit Task 2.

The encoder consists of 3 parts:

  1. We extract ngram features from character embeddings. This is realised by
     1d-convolutions over the embedded sequences with filter width of size
     $n$. All ngram features are concatenated and projected to a hidden
     dimension.

  2. To get a single embedding for each word, we run a BiLSTM on the ngram
     features and concatenate the last hidden states (forward and backward).

  3. To condition the word embeddings on context, we run a BiLSTm on the word
     embeddings. In order not to loose character information, we concatenate
     the non-contextual embeddings to the contextual embeddings
     (skip-connection) and project to a lower dimension.
"""


import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence

from char2token import AvgPoolChar2TokenEncoder
from char2token import MaxPoolChar2TokenEncoder
from char2token import LSTMChar2TokenEncoder

from mlp import LSTM, ResidualMLP, mlp


class Classifier(nn.Module):
    """
    Complete encoder:
    Combines convolutional ngram encoders, LSTM word encoder, and LSTM
    context encoder into a single module.

    Returns:
    Token embeddings ( = encodings)
    Projected ngram embeddings (used as input to decoder attention)
    Length of all tokens (in characters)
    """

    def __init__(
        self,
        hidden_dim,
        num_stem_classes,
        num_tag_classes,
        char2token_mode="rnn",
        dropout=0.0,
    ):
        super().__init__()

        if hidden_dim % 2 != 0:
            raise ValueError("Hidden dim can only be even number!")

        # Instantiate char embeddings -> token embedding converter
        if char2token_mode == "rnn":
            self.char2token = LSTMChar2TokenEncoder(
                hidden_dim, hidden_dim=hidden_dim // 2, dropout=dropout
            )
        elif char2token_mode == "avg":
            self.char2token = AvgPoolChar2TokenEncoder()
        elif char2token_mode == "max":
            self.char2token = MaxPoolChar2TokenEncoder()
        else:
            raise ValueError(f"Unknown char -> token converter: {char2token_mode}")

        self.transform = ResidualMLP(hidden_dim, 1, dropout=dropout)
        self.stem_rule_classifier = mlp(
            hidden_dim, hidden_dim, 2, output_dim=num_stem_classes, dropout=dropout
        )
        self.tag_classifier = mlp(
            hidden_dim, hidden_dim, 2, output_dim=num_tag_classes, dropout=dropout
        )

    @staticmethod
    def convert_chars_to_tokens(char_embeddings, boundaries):
        # char_embeddings: shape (batch, #chars, features)
        assert len(char_embeddings) == len(boundaries)

        tokens = []
        token_lengths = []
        for sentence, token_boundaries in zip(char_embeddings, boundaries):
            for start, stop in token_boundaries:
                token = sentence[start:stop]
                tokens.append(token)
                token_lengths.append(len(token))

        char_embeddings = pad_sequence(tokens, batch_first=True)
        # shape (batch, #chars, features)
        token_lengths = torch.LongTensor(token_lengths).to(char_embeddings.device)

        return char_embeddings, token_lengths

    def forward(self, char_embeddings, boundaries):
        # char_embeddings: shape (batch, #chars, features)
        char_embeddings, token_lengths = self.convert_chars_to_tokens(
            char_embeddings, boundaries
        )
        # shape (#token, #chars, #features)

        # Calculate token embeddings
        tokens = self.char2token(char_embeddings, token_lengths)
        tokens = self.transform(tokens)
        # shape (#tokens, features)

        # Calculate prediction scores
        y_pred_stem = self.stem_rule_classifier(tokens)
        y_pred_tag = self.tag_classifier(tokens)

        return y_pred_stem, y_pred_tag
