"""
Our segmentation model consists of 3 parts:
  1. The input chars are encoded by char embeddings. From the char embeddings,
     we extract ngram features by applying 1d-convolutions of different filter
     sizes. For each ngram length, we have multiple 1d-convolution layers, but
     the first 1d-convolution layer is added to the final features via a
     skip-connection.
  2. The different ngram features (of each char) are concatenated and
     projected to a lower dimension. The resulting char features are further
     encoded by a 2-layer BiLSTM. This is to condition predictions of
     Sandhi rules on more context.
     Note, however, that, given the general shortness of sequences, the
     receptive field of convolutions for larger ngram lengths (e.g. 7)
     should be already large enough to see most relevant context.
  3. We predict a Sandhi rule for each char using a MLP classifier with the
     output of the BiLSTM as input. Furthermore, we apply layer norm to the
     outputs of the BiLSTM.
"""


import torch
import torch.nn as nn

# "mlc" stands for "multi-layer convolution"
# (cf. mlp = multi-layer perceptron)
from mlp import mlc, mlp, LSTM


class SingleNgramEncoder(nn.Module):
    """
    Implements a stack of 1d-convolutions + skip connection with given
    filter size (`ngram`).
    Extracts ngram features from char embeddings.
    
    input_dim: Embeddings size
    ngram:     Filter size
    """

    def __init__(self, input_dim, hidden_dim, ngram, dropout=0.0):
        super().__init__()

        # First layer 1d-conv
        self.conv1d = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv1d(input_dim, hidden_dim, ngram, padding="same"),
            nn.ReLU(),
        )

        # Rest stack of 1d-convolutions
        self.transform = mlc(hidden_dim, hidden_dim, ngram, 2, dropout=dropout)

    def forward(self, char_embeddings):
        # char_embeddings: shape (batch, #chars, features)
        #
        # To apply our convolutional ngram encoders, we have to flip
        # dimensions:
        char_embeddings = char_embeddings.transpose(-1, -2)
        # shape (batch, features, #chars)
        ngram_embeddings = self.conv1d(char_embeddings)
        # shape (batch, hidden, #chars)

        # Apply nonlinear transform (more convolutions for context):
        transformed = self.transform(ngram_embeddings)
        # shape (batch, hidden, #chars)

        # Skip connection:
        ngram_embeddings = ngram_embeddings + transformed
        # shape (batch, hidden, #chars)

        # Reverse flipping of dimensions:
        ngram_embeddings = ngram_embeddings.transpose(-1, -2)
        # shape (batch * #tokens, #chars, hidden)

        return ngram_embeddings


class SegmenterModel(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        num_classes,
        embedding_dim,
        hidden_dim,
        max_ngram,
        dropout=0.0,
    ):
        super().__init__()
        assert (
            hidden_dim % 2 == 0
        ), f"Hidden dim must be divisible by 2, but is {hidden_dim}"

        # Char embedding matrix
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)

        # Convolutional ngram feature extractors
        self.ngram_encoders = [
            SingleNgramEncoder(embedding_dim, hidden_dim, ngram, dropout=dropout)
            for ngram in range(2, max_ngram + 1)
        ]
        self.ngram_encoders = nn.ModuleList(self.ngram_encoders)

        # Ngram feature downsampling
        self.ngram_downsample = mlp(
            (max_ngram - 1) * hidden_dim, hidden_dim, 1, dropout=dropout
        )

        # BiLSTM for context sensivity
        self.lstm = LSTM(hidden_dim, hidden_dim // 2, num_layers=2, dropout=dropout)
        self.layer_norm = nn.LayerNorm((hidden_dim,))

        # Classification layer
        self.predictions = mlp(
            hidden_dim, hidden_dim, 1, output_dim=num_classes, dropout=dropout
        )

    def forward(self, source):
        # source: shape (batch, #chars)
        lengths = (source != 0).sum(dim=-1).flatten().long()

        # Convert char indices -> embeddings
        char_embeddings = self.embedding(source)
        # shape (batch, chars, features)

        # Extract ngram features by 1d-convolutions
        ngram_embeddings = [encoder(char_embeddings) for encoder in self.ngram_encoders]

        # Combine & downsample
        ngram_embeddings = torch.cat(ngram_embeddings, dim=-1)
        char_embeddings = self.ngram_downsample(ngram_embeddings)
        # shape (batch, #chars, features)

        # Run BiLSTM
        transformed = self.lstm(char_embeddings, lengths)
        # Skip connection: Bypass LSTM
        char_embeddings = char_embeddings + transformed
        # Layer norm: Normalise char features
        char_embeddings = self.layer_norm(char_embeddings)

        # Predict unnormalised prediction scores
        # -> use Cross Entropy Loss later
        scores = self.predictions(char_embeddings)

        return scores
