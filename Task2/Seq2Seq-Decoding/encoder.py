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

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from conv_ngram_encoder import ConvNgramEncoder
from char2token import AvgPoolChar2TokenEncoder
from char2token import MaxPoolChar2TokenEncoder
from char2token import LSTMChar2TokenEncoder

from mlp import LSTM, ResidualMLP, mlp


class Encoder(nn.Module):
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
        vocabulary_size,
        embedding_dim,
        hidden_dim,
        ngram,
        char2token_mode="rnn",
        dropout=0.0,
    ):
        super().__init__()

        if hidden_dim % 2 != 0:
            raise ValueError("Hidden dim can only be even number!")

        self.output_size = hidden_dim

        # Instantiate embedding matrix
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)

        # Instantiate char (ngram) encoder
        self.char_encoder = ConvNgramEncoder(
            embedding_dim, hidden_dim, ngram, dropout=dropout
        )

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

        # Instantiate module that conditions the character based embeddings
        # on context
        self.context_encoder = LSTM(
            hidden_dim, hidden_dim, num_layers=2, dropout=dropout
        )

        # Instantiate nonlinear downsampling of combined contextual and
        # noncontextual token embeddings
        self.downsample = mlp(3 * hidden_dim, hidden_dim, 2, dropout=dropout)

        # Instantiate final projection
        self.projection = ResidualMLP(hidden_dim, 2, dropout=dropout)

    def forward(self, inputs):
        # inputs: shape (batch, #tokens, #chars)
        batch, num_tokens, num_chars = inputs.shape

        # First, we calculate token and sentence lengths:
        token_lengths = (inputs != 0).sum(dim=-1).long()  # shape (batch, #tokens)
        sent_lengths = (token_lengths != 0).sum(dim=-1).long()

        token_indices = token_lengths.flatten().nonzero().flatten()

        # Embed inputs (chars)
        char_embeddings = self.embedding(inputs)
        # shape (batch, #tokens, #chars, features)

        # Calculate ngram features
        char_embeddings = torch.flatten(char_embeddings, end_dim=-3)
        char_embeddings = self.char_encoder(char_embeddings)
        # shape (batch * #tokens, #chars, features)

        # Calculate token embeddings
        noncontextual_embeddings = self.char2token(char_embeddings, token_lengths)
        # shape (batch * #tokens, features)

        # Calculate contextual token embeddings
        contextual_embeddings = noncontextual_embeddings.reshape(batch, num_tokens, -1)
        contextual_embeddings = self.context_encoder(
            contextual_embeddings, sent_lengths
        )
        contextual_embeddings = torch.flatten(contextual_embeddings, end_dim=-2)
        # shape (batch * #tokens, features)

        # Concatenate contextual and non-contextual token embeddings
        token_embeddings = [noncontextual_embeddings, contextual_embeddings]
        token_embeddings = torch.cat(token_embeddings, dim=-1)

        # Remove tokens that correspond to padding
        token_lengths = token_lengths.flatten()[token_indices].flatten().contiguous()
        char_embeddings = char_embeddings[token_indices].contiguous()
        token_embeddings = token_embeddings[token_indices].contiguous()
        noncontextual_embeddings = noncontextual_embeddings[token_indices].contiguous()

        # Downsample combibned contextual and non-contextual token embeddings
        token_embeddings = self.downsample(token_embeddings)

        # Add non contextual token embeddings to downsampled embeddings
        # (residual connection)
        token_embeddings = token_embeddings + noncontextual_embeddings

        # Final projection of token embeddings
        token_embeddings = self.projection(token_embeddings)

        return token_embeddings, char_embeddings, token_lengths
