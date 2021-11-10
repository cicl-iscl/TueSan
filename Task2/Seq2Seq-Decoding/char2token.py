"""
Implement char to token encoders:
Take sequence of char embeddings as input
    shape (batch = #tokens, #chars, features)
Output 1 token embedding
    shape (batch = #tokens, features)
    
Implemented variants:
 1. BiLSTM - Concatenate last hidden states of forward / backward LSTM
 2. Average Pooling
 3. Max Pooling
"""

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from mlp import mlp


def lengths2mask(lengths, feature_dim=None):
    with torch.no_grad():
        # Start with sequence of ascending integers
        mask = torch.arange(torch.max(lengths), device=lengths.device)
        # Replicate along batch dim
        mask = mask.unsqueeze(0)
        mask = mask.tile(lengths.shape[0], 1)
        # For each entry, first `length` indices are < `lengths
        # -> turn to 1, others 0
        mask = mask >= lengths.unsqueeze(-1)

        if feature_dim is not None:
            mask = mask.unsqueeze(-1)
            mask = mask.tile(1, 1, feature_dim)

        mask = mask.bool()
        return mask


class AvgPoolChar2TokenEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, char_embeddings, token_lengths):
        # char_embeddings: shape (#tokens, #chars, features)
        # token_lengths: (#tokens,)

        # Flatten token lengths
        token_lengths = torch.flatten(token_lengths)
        assert len(token_lengths) == char_embeddings.shape[0]

        # Create mask
        embedding_dim = char_embeddings.shape[-1]
        mask = lengths2mask(token_lengths, feature_dim=embedding_dim)

        # Replace padded values with 0.0
        char_embeddings = torch.masked_fill(char_embeddings, mask, 0.0)

        # Sum along char dimension (feature-wise pooling)
        token_embeddings = char_embeddings.sum(dim=1)
        # shape (batch, features)

        # Devide by sequence length (= take average)
        # First, we have to clamp `lengths` to 1 -> avoid ZeroDivisionError
        token_lengths = torch.clamp(token_lengths, min=1)
        token_lengths = token_lengths.unsqueeze(-1)
        token_embeddings = token_embeddings / token_lengths

        return token_embeddings


class MaxPoolChar2TokenEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.mask_value = torch.finfo().min

    def forward(self, char_embeddings, token_lengths):
        # char_embeddings: shape (#tokens, #chars, features)
        # token_lengths: (#tokens,)

        # Flatten token lengths
        token_lengths = torch.flatten(token_lengths)
        assert len(token_lengths) == char_embeddings.shape[0]

        # Create mask
        embedding_dim = char_embeddings.shape[-1]
        mask = lengths2mask(token_lengths, feature_dim=embedding_dim)

        # Replace padded values with 0.0
        char_embeddings = char_embeddings.masked_fill(mask, self.mask_value)

        # Sum max char dimension (feature-wise pooling)
        token_embeddings = torch.amax(char_embeddings, dim=1)

        # shape (batch, features)

        return token_embeddings


class LSTMChar2TokenEncoder(nn.Module):
    """
    Apply BiLSTM to character features
    -> concat last hidden states
    -> word embedding
    """

    def __init__(
        self, input_dim: int, hidden_dim=None, dropout=0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else input_dim

        self.hidden_dim = hidden_dim
        # Instantiate LSTM
        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

    def forward(self, char_embeddings, token_lengths):
        # Taken from https://github.com/pytorch/pytorch/issues/4582#issuecomment-589905631
        # PyTorch can't handle 0 length sequences, so we have to waste some
        # computation
        # In particular, we force pytorch to assume all tokens actually
        # have at least 1 character and overwrite padded tokens later
        #
        # Set minimum length = 1
        token_lengths = torch.flatten(token_lengths).cpu()
        clamped_lengths = torch.clamp(token_lengths, min=1)
        # Pack sequences
        char_embeddings = pack_padded_sequence(
            char_embeddings, clamped_lengths, batch_first=True, enforce_sorted=False
        )
        # Apply BiLSTM
        _, (hidden, _) = self.rnn(char_embeddings)
        # Do some dimension housekeeping (not important)
        hidden = hidden.reshape(2, -1, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.reshape(-1, 2 * self.hidden_dim)

        # At the end, we mask the word embeddings that correspond to
        # padding (padded tokens) by replacing them with 0s
        mask = token_lengths == 0  # Find padded tokens
        mask = mask.reshape(-1, 1)  # Represent mask as 1 list of tokens
        mask = mask.to(hidden.device)
        token_embeddings = torch.masked_fill(hidden, mask, 0.0)

        return token_embeddings
