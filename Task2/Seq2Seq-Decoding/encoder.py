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


def conv_ngram_encoder_factory(
    input_dim: int, ngram: int, hidden_dim: int, output_dim: int, dropout=0.0
):
    """
    Create convolutional ngram encoder.
    """
    model = nn.Sequential(
        # Extract ngram features by 1d convolution
        nn.Conv1d(input_dim, hidden_dim, ngram, padding="same"),
        nn.Dropout(p=dropout),
        nn.ReLU(),
        # Nonlinear transform on top of ngram features.
        nn.Conv1d(hidden_dim, hidden_dim, 1, padding="same"),
        nn.Dropout(p=dropout),
        nn.ReLU(),
        nn.Conv1d(hidden_dim, output_dim, 1, padding="same"),
        nn.Dropout(p=dropout),
        nn.ReLU(),
    )

    return model


class RecurrentWordEncoder(nn.Module):
    """
    Apply BiLSTM to character features
    -> concat last hidden states
    -> word embedding
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout=0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # Instantiate LSTM
        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        # Instantiate projection MLP
        self.projection = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )

    def forward(self, input, lengths):
        # Taken from https://github.com/pytorch/pytorch/issues/4582#issuecomment-589905631
        # PyTorch can't handle 0 length sequences, so we have to waste some
        # computation
        # In particular, we force pytorch to assume all tokens actually
        # have at least 1 character and overwrite padded tokens later
        #
        # Set minimum length = 1
        clamped_lengths = torch.clamp(lengths, min=1)
        # Pack sequences
        input = pack_padded_sequence(
            input, clamped_lengths, batch_first=True, enforce_sorted=False
        )
        # Apply BiLSTM
        _, (hidden, _) = self.rnn(input)
        # Do some dimension housekeeping (not important)
        hidden = hidden.reshape(2, -1, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.reshape(-1, 2 * self.hidden_dim)

        # Project to output dimensionality and apply nonlinearity
        # embedding shape: (batch, features)
        embedding = self.projection(hidden)

        # At the end, we mask the word embeddings that correspond to
        # padding (padded tokens) by replacing them with 0s
        mask = lengths == 0  # Find padded tokens
        mask = mask.reshape(-1, 1)  # Represent mask as 1 list of tokens
        mask = mask.to(embedding.device)
        embedding = torch.masked_fill(embedding, mask, 0.0)

        return embedding


class Char2TokenEncoder(nn.Module):
    """
    Extract character ngram features and calculate word embeddings
    from raw sentences.
    """

    def __init__(
        self, vocabulary_size, embedding_dim, hidden_dim, max_ngram, dropout=0.0
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.ngram_encoder_output_dim = hidden_dim
        self.ngram_encoder_hidden_dim = hidden_dim
        self.rnn_word_encoder_hidden_dim = hidden_dim
        self.rnn_word_encoder_output_dim = hidden_dim

        # Embedding matrix
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        # Instantiate convolutional
        ngram_range = range(1, max_ngram + 1)
        ngram_encoders = [
            conv_ngram_encoder_factory(
                embedding_dim,
                n,
                self.ngram_encoder_hidden_dim,
                self.ngram_encoder_output_dim,
                dropout=dropout,
            )
            for n in iter(ngram_range)
        ]
        self.ngram_encoders = nn.ModuleList(ngram_encoders)
        # Output dim of convolutional ngram encoder is
        # max ngram length * hidden dim
        self.ngram_output_dim = len(list(iter(ngram_range)))
        self.ngram_output_dim *= self.ngram_encoder_output_dim

        # Instantiate token encoder
        self.word_encoder = RecurrentWordEncoder(
            self.ngram_output_dim,
            self.rnn_word_encoder_hidden_dim,
            self.rnn_word_encoder_output_dim,
            dropout=dropout,
        )

        # Instantiate ngram embedding downsampling
        # (used for preparing convolutional ngram embeddings
        #  as input to decoder attention)
        self.ngram_down_sample = nn.Sequential(
            nn.Linear(self.ngram_output_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, inputs):
        # inputs is the input sentence
        # inputs: shape (batch, #tokens, #chars)
        batch, num_tokens, num_chars = inputs.shape

        token_lengths = (inputs != 0).cpu()  # shape (batch, #tokens, #chars)
        token_lengths = token_lengths.sum(dim=-1)  # shape (batch, #tokens)
        token_lengths = token_lengths.reshape(-1)

        # Process each word alone:
        #
        # First, we embed each character
        #
        inputs = self.embedding(inputs)
        # shape (batch, #tokens, #chars, features)
        #
        # Next, we reshape our tensor s.t. each batch element corresponds to
        # exactly 1 token:
        inputs = inputs.reshape(batch * num_tokens, num_chars, self.embedding_dim)
        # shape (batch * #tokens, #chars, features)

        # To apply our convolutional ngram encoders, we have to flip
        # dimensions:
        inputs = inputs.transpose(-1, -2)
        # shape (batch * #tokens, features, #chars)

        ngram_embeddings = [
            ngram_encoder(inputs) for ngram_encoder in self.ngram_encoders
        ]
        ngram_embeddings = torch.cat(ngram_embeddings, dim=-2)
        # shape (batch * #tokens, features, #chars)

        ngram_embeddings = ngram_embeddings.transpose(-1, -2)
        # shape (batch * #tokens, #chars, features)

        # Finally, we encode each token by the concatenation of last forward
        # and backward hidden states of a BiLSTM applied to the ngram
        # features:
        token_embeddings = self.word_encoder(ngram_embeddings, token_lengths)
        # shape (batch * #tokens, features)

        token_embeddings = token_embeddings.reshape(batch, num_tokens, -1)
        # shape (batch, #tokens, features)

        token_lengths = token_lengths.reshape(batch, num_tokens)
        sent_lengths = (token_lengths != 0).sum(dim=-1).reshape(-1)

        # For the decoder, we later also need char embeddings
        # so we calculate them here
        char_embeddings = self.ngram_down_sample(ngram_embeddings)
        token_lengths = token_lengths.reshape(-1)

        return token_embeddings, char_embeddings, sent_lengths, token_lengths


class RNNSentenceEncoder(nn.Module):
    """
    Apply BiLSTM to word embeddings
    -> contextual embeddings, that is, embeddings of individual words
       are conditioned on the context. This introduces semantic information
       that can potentially help to disambiguate stems/morphological tags
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Instantiate LSTM
        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Instantiate MLP projection
        self.projection = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )

    def forward(self, inputs, lengths):
        # Pack sentences
        inputs = pack_padded_sequence(
            inputs, lengths, batch_first=True, enforce_sorted=False
        )
        # Apply LSTM
        inputs, _ = self.rnn(inputs)
        # Unpack sentences
        inputs, lengths = pad_packed_sequence(inputs, batch_first=True)

        # Project to output dimensionality and apply nonlinearity
        projection = self.projection(inputs)
        return projection


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
        self, vocabulary_size, embedding_dim, hidden_dim, max_ngram, dropout=0.0
    ):
        super().__init__()
        self.output_size = hidden_dim

        # Instantiate module that calculates 1 character based embedding
        # for each token
        self.char2token_encoder = Char2TokenEncoder(
            vocabulary_size, embedding_dim, hidden_dim, max_ngram, dropout=dropout
        )

        # Instantiate module that conditions the character based embeddings
        # on context
        self.contextual_encoder = RNNSentenceEncoder(
            hidden_dim, hidden_dim, hidden_dim, dropout=dropout
        )

        # Instantiate MLP for projecting the combined contextual and
        # character based embeddings
        self.projection = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )

    def forward(self, inputs):
        # Calculate character based word embeddings
        (
            token_embeddings,
            char_embeddings,
            sent_lengths,
            token_lengths,
        ) = self.char2token_encoder(inputs)

        # Calculate contextual word embeddings
        contextual_embeddings = self.contextual_encoder(token_embeddings, sent_lengths)

        # Concatenate contextual and non-contextual embeddings
        # We do this to ensure that character information is not
        # overwritten by the contextual encoder
        encodings = torch.cat([token_embeddings, contextual_embeddings], dim=-1)

        # Project and apply nonlinearity
        encodings = self.projection(encodings)

        return encodings, char_embeddings, token_lengths
