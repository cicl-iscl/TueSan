"""
Implements the training loop
"""

import torch

from tqdm import tqdm


def train(model, criterion, optimizer, dataloader, epochs, device):

    running_loss = None
    for epoch in range(epochs):
        batches = tqdm(dataloader)
        for inputs, labels in batches:
            y_pred = model(inputs.to(device))

            optimizer.zero_grad()
            loss = criterion(y_pred, labels.to(device))
            loss.backward()
            optimizer.step()

            detached_loss = loss.detach().cpu().item()
            if running_loss is None:
                running_loss = detached_loss
            else:
                running_loss = 0.95 * running_loss + 0.05 * detached_loss

            batches.set_postfix_str(
                "Loss: {:.2f}, Running Loss: {:.2f}".format(
                    detached_loss * 100, running_loss * 100
                )
            )

    return model, optimizer
