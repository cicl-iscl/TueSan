"""
Implements the training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


def train(model, optimizer, criterion, dataloader, epochs, device):
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    model = model.to(device)

    running_loss = None
    for epoch in range(epochs):
        batches = tqdm(dataloader)

        for inputs, labels in batches:
            y_pred = model(inputs.to(device))

            y_pred = y_pred.flatten(end_dim=-2)
            labels = labels.flatten().long().to(device)

            optimizer.zero_grad()
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()

            detached_loss = loss.detach().cpu().item()
            if running_loss is None:
                running_loss = detached_loss
            else:
                running_loss = 0.95 * running_loss + 0.05 * detached_loss

            batches.set_postfix_str(
                "Loss: {:.2f}, Running Loss: {:.2f}".format(detached_loss, running_loss)
            )

    return model, optimizer
