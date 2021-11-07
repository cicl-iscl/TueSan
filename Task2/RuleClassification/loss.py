"""
### Loss

We need a modified version of the cross-entropy loss where padded
characters/words don't contribute to the loss. This is really only a technical
trick because the standard PyTorch implementation of cross-entropy loss
doesn't allow sequence masking.
"""


import torch
import torch.nn.functional as F


def masked_cross_entropy_loss(y_pred, y_true, length):
    # y_pred: shape (batch, timesteps, classes)
    # y_true: shape (batch, timesteps)
    # length: shape (batch,)

    # Make binary mask: shape (batch, timesteps)
    # The mask tells us which tokens are padding
    # Cf. https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
    batch_size = length.shape[0]
    max_len = torch.max(length)
    seq_range = torch.arange(0, max_len).long().to(y_pred.device)
    seq_range_expand = seq_range.unsqueeze(0)
    seq_range_expand = seq_range_expand.expand(batch_size, y_pred.shape[1])
    seq_length_expand = length.unsqueeze(1)
    seq_length_expand = seq_length_expand.expand_as(seq_range_expand)
    mask = seq_range_expand < seq_length_expand

    # Reshape:
    # We want 2 dimensions: First token, then prediction probabilities
    # for each token
    num_classes = y_pred.shape[2]
    y_pred = y_pred.reshape(-1, num_classes)
    y_true = y_true.reshape(-1)
    mask = mask.reshape(-1)

    # Calculate loss
    # We calculate cross-entropy loss for each batch element
    loss = F.cross_entropy(y_pred, y_true, reduction="none")
    # We only keep losses for batch elements that are NOT padding
    loss = torch.masked_select(loss, mask)
    # Normalise loss
    loss = loss.mean()

    return loss
