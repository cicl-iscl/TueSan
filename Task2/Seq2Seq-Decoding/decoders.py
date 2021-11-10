"""
Implements the character-based decoder LSTM for predicting stems
and the tag classifier.
"""

import torch
import torch.nn as nn

from attention import BahdanauAttention


def mlp_classifier_factory(input_dim, hidden_dim, num_classes, layers=1, dropout=0.0):
    """
    Implements a MLP as a single nn.Sequential module
    containing num_layers x [linear, dropout, nonlinearity]
    and eventually calculates prediction scores for each class.
    
    We do not apply a final activation function like softmax or
    log-softmax, because we use PyTorch's cross-entropy loss
    which operates on un-normalised scores.
    """
    modules = []
    # Add layers
    for layer in range(layers + 1):
        # Input dimensionality is input_dim for first layer
        # else hidden dim
        dim_in = input_dim if layer == 0 else hidden_dim
        # Output dim is hidden dim, except for the last layer
        # where we need num_classes many output values
        dim_out = num_classes if layer == layers else hidden_dim

        # Add linear transform and dropout
        modules.append(nn.Linear(dim_in, dim_out))
        modules.append(nn.Dropout(dropout))

        # Don't add nonlinearity for last ( = prediction) layer
        if layer < layers:
            modules.append(nn.ReLU())

    return nn.Sequential(*modules)


class RecurrentDecoder(nn.Module):
    """
    Implements a LSTM decoder that predicts 1 character each timestep.
    """

    def __init__(
        self,
        vocab_size,
        embedding_size,
        encoder_output_size,
        hidden_size,
        num_layers,
        dropout=0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Instantiate embedding matrix
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Instantiate decoder RNN
        # We give the token embedding and the embedding of the
        # previously predicted character as inputs to the decoder
        input_size = embedding_size + encoder_output_size
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )

        # Instantiate attention module
        self.attention = BahdanauAttention(
            encoder_output_size, hidden_size, hidden_size, dropout=dropout
        )

        # Instantiate MLP for making predictions
        self.prediction_layer = nn.Sequential(
            nn.Linear(encoder_output_size + hidden_size, hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )

    def _forward_step(
        self,
        last_output,
        encoder_output,
        encoder_char_embeddings,
        encoder_char_lengths,
        hidden=None,
    ):
        """Predict next character (single prediction step)"""
        # last_output: shape (batch,)
        # encoder_output: shape (batch, features)
        # hidden: shape (num_layers, batch, features)

        # Embed predicted character
        last_embedding = self.embedding(last_output)
        # Concatenate to encoder word embedding
        last_embedding = torch.cat([last_embedding, encoder_output], dim=-1)
        last_embedding = last_embedding.unsqueeze(1)

        # Apply LSTM
        output, hidden = self.rnn(last_embedding, hidden)
        # output: shape (batch, 1, features)
        output = output.squeeze(1)

        # Calculate attention (context vectors) and concatenate to output
        attention_hidden = hidden[0][-1].contiguous()
        context_vectors = self.attention(
            encoder_char_embeddings, attention_hidden, encoder_char_lengths
        )
        output = torch.cat([output, context_vectors], dim=-1)

        # Calculate classification scores
        output = self.prediction_layer(output)
        return output, hidden

    def forward(
        self, targets, encoder_output, encoder_char_embeddings, encoder_char_lengths
    ):
        """
        Only used for training:
        Predict characters by teacher forcing
        """
        # targets: shape (batch, #chars)
        # encoder_output: shape (batch, features)

        outputs = []
        hidden = None
        num_chars = targets.shape[1]

        for i in range(num_chars):
            last_output = targets[:, i]
            output, hidden = self._forward_step(
                last_output,
                encoder_output,
                encoder_char_embeddings,
                encoder_char_lengths,
                hidden=hidden,
            )
            outputs.append(output)

        outputs = torch.stack(outputs)
        outputs = outputs.transpose(0, 1)

        return outputs
