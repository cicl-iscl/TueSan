"""
Implements MLP (Bahdanau-style) attention.

Given the encoder hidden state and the character embeddings of the input token,
calculate a so-called attention score for each input character.
Then, attention scores are normalised across all input characters.
Finally, we represent the input encoding at the current timestep as a
weighted sum of character embeddings (weights = attention scores)

The idea is as follows:
Whenever we predict the next character, we dynamically calculate which
part of the input is currently most important. Thereby, we get a different
encoding of the input for each predicted character instead of only a fixed
token embedding. This makes the model more flexible.

The term attention comes from the idea that the model focuses ('attends') on
the currently most relevant part of the input.

For predicting stems, this could be helpful, if the stem is (partially)
present in the input, because then the decoder can simply pick out the
currently matching input characters.

Another advantage of attention is in terms of information flow:
Using attention introduces a shortcut connection how input information
(here from the convolutional ngram encoders) influences the decoder output.
This may speed up optimisation, because all intermediate representations are
bypassed and the prediction errors during training are backpropagated much
more directly to the input encodings.

MLP attention employs a MLP to calculate the attention score
from the character embeddings and the decoder hidden state.
There are also other flavours of attention, for example bilinear attention
and self-attention.
"""

import torch
import torch.nn as nn

import numpy as np


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size, hidden_size, dropout=0.0):
        super().__init__()

        # Instantiate MLP for calculating attention scores
        self.attention = nn.Sequential(
            nn.Linear(encoder_size + decoder_size, hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Instantiate softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoder_out, decoder_hidden, encoder_length):
        # encoder_out: shape (batch, timesteps, features)
        # decoder_hidden: shape (batch, features)
        # encoder_length: shape (batch,)

        # First, we have to duplicate the decoder hidden state
        # along the `timesteps` dimension of the encoder
        # And concatenate a copy of the decoder hidden state to each
        # character embedding
        batch, timesteps = encoder_out.shape[:2]
        # Make `timesteps` dim
        decoder_hidden = decoder_hidden.unsqueeze(1)
        # Make copies of decoder hidden state along `timesteps` dim
        decoder_hidden = decoder_hidden.tile(1, timesteps, 1)
        # Concatenate to character embeddings
        attention_input = torch.cat([encoder_out, decoder_hidden], dim=-1)

        # Calculate attention scores by applying attention MLP
        scores = self.attention(attention_input)
        # Get scores for each character (timesteps) for each token in batch
        scores = scores.reshape(batch, timesteps)

        # Next, we have to normalise attention scores
        # for each token, that is, they should sum to 1
        # One problem here is that we have padding, so padded
        # characters should receive 0 attention. This can be
        # achieved by replacing their unnormalized attention score
        # with -infinity. Then it will be 0 after applying softmax.
        #
        # To handle padded characters, we first have to calculate a mask
        # that tells us which characters are padded and which not.
        #
        # Convert lengths to binary mask
        # From https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
        mask = torch.arange(timesteps)
        mask = mask.expand(len(encoder_length), timesteps)
        mask = mask.to(encoder_length.device)
        mask = mask >= encoder_length.unsqueeze(1)
        mask = mask.to(encoder_out.device)

        # Mask scores that correspond to padding
        # From https://github.com/joeynmt/joeynmt/blob/master/joeynmt/attention.py
        scores = torch.masked_fill(scores, mask, -np.inf)
        # Apply softmax
        scores = self.softmax(scores)
        # This will lead to errors for padded tokens (that have 0 characters)
        # because softmax can't handle this case.
        # Therefore, we have to again mask scores that correspond to padding,
        # this time replacing them with 0.
        # The double masking is necessary to avoid incorrect normalisation
        # when applying softmax.
        # In practise, we could also use a very low number like -100 instead
        # of -infinity and only mask once, but conceptionally it is cleaner
        # this way, I think
        scores = torch.masked_fill(scores, mask, 0.0)
        # Remove trailing score dimension
        scores = scores.unsqueeze(2)

        # Apply batch matrix multiplication to calculate
        # weighted sums of input character emmbeddings
        encoder_out = encoder_out.transpose(1, 2)
        context_vectors = torch.bmm(encoder_out, scores)
        context_vectors = context_vectors.squeeze(2)

        return context_vectors
