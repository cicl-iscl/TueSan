import torch
import torch.nn as nn

from mlp import ResidualMLP


class ConvNgramEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, ngram, dropout=0.0):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.convolutions = [
            self.make_1d_convolution(ngram) for ngram in range(1, ngram + 1)
        ]
        self.convolutions = nn.ModuleList(self.convolutions)
        self.downsample = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(ngram * hidden_dim, hidden_dim), nn.ReLU()
        )
        self.transform = ResidualMLP(
            hidden_dim, 2, hidden_dim=hidden_dim, dropout=dropout
        )

    def make_1d_convolution(self, ngram):
        return nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Conv1d(self.embedding_dim, self.hidden_dim, ngram, padding="same"),
            nn.ReLU(),
        )

    def forward(self, char_embeddings):
        # char_embeddings: shape (batch * #tokens, #chars, features)
        #
        # To apply our convolutional ngram encoders, we have to flip
        # dimensions:
        char_embeddings = char_embeddings.transpose(-1, -2)
        # shape (batch * #tokens, features, #chars)

        ngram_embeddings = [conv1d(char_embeddings) for conv1d in self.convolutions]
        ngram_embeddings = torch.cat(ngram_embeddings, dim=1)
        # shape (batch * #tokens, hidden, #chars)

        # Reverse flipping of dimensions:
        ngram_embeddings = ngram_embeddings.transpose(-1, -2)
        # shape (batch * #tokens, #chars, hidden)

        # Apply nonlinear transform:
        transformed = self.downsample(ngram_embeddings)
        transformed = self.transform(transformed)
        # shape (batch * #tokens, #chars, hidden)

        return transformed
