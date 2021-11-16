import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp import mlc, mlp, LSTM


class SingleNgramEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, ngram, dropout=0.0):
        super().__init__()

        self.conv1d = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv1d(input_dim, hidden_dim, ngram, padding="same"),
            nn.ReLU(),
        )

        self.transform1 = mlc(hidden_dim, hidden_dim, ngram, 2, dropout=dropout)
        # self.transform2 = mlc(hidden_dim, hidden_dim, ngram, 4, dropout=dropout)

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
        transformed = self.transform1(ngram_embeddings)
        # transformed2 = self.transform2(ngram_embeddings)
        # transformed = torch.stack([transformed1, transformed2])
        # transformed = torch.amax(transformed, dim = 0)
        # shape (batch, hidden, #chars)

        # Skip connection:
        ngram_embeddings = ngram_embeddings + transformed
        # shape (batch, hidden, #chars)

        # Reverse flipping of dimensions:
        ngram_embeddings = ngram_embeddings.transpose(-1, -2)
        # shape (batch * #tokens, #chars, hidden)

        return ngram_embeddings


class Encoder(nn.Module):
    def __init__(
        self, vocabulary_size, embedding_dim, hidden_dim, max_ngram, dropout=0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.ngram_encoders = [
            SingleNgramEncoder(embedding_dim, hidden_dim, ngram, dropout=dropout)
            for ngram in range(2, max_ngram + 1)
        ]
        self.ngram_encoders = nn.ModuleList(self.ngram_encoders)
        self.ngram_downsample = mlp(
            (max_ngram - 1) * hidden_dim, hidden_dim, 1, dropout=dropout
        )
        self.transform = mlp(hidden_dim, hidden_dim, 2, dropout=dropout)

    def forward(self, source):
        # source: shape (batch, chars)
        lengths = (source != 0).sum(dim=-1).flatten().long()

        char_embeddings = self.embedding(source)
        # shape (batch, chars, features)
        ngram_embeddings = [encoder(char_embeddings) for encoder in self.ngram_encoders]
        ngram_embeddings = torch.cat(ngram_embeddings, dim=-1)
        ngram_embeddings = self.ngram_downsample(ngram_embeddings)

        transformed = self.transform(ngram_embeddings)
        ngram_embeddings = ngram_embeddings + transformed

        return ngram_embeddings
