"""
Implements the training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


def train(model, optimizer, criterion, dataloader, epochs, device):
    model = model.to(device)

    # Use some fancy learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, epochs=epochs, steps_per_epoch=len(dataloader)
    )

    # Mainting running average of loss
    running_loss = None

    # Train for given number of epochs
    for epoch in range(epochs):
        batches = tqdm(dataloader, desc=f"Epoch: {epoch}")

        # Iterate over minibatches
        for inputs, labels in batches:
            # Calculate predictions
            y_pred = model(inputs.to(device))

            # Flatten: Chars in rows and class scores in columns
            y_pred = y_pred.flatten(end_dim=-2)
            labels = labels.flatten().long().to(device)

            # Update model
            optimizer.zero_grad()
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Display loss & learning rate
            detached_loss = loss.detach().cpu().item()
            lr = scheduler.get_last_lr()[0]
            if running_loss is None:
                running_loss = detached_loss
            else:
                running_loss = 0.95 * running_loss + 0.05 * detached_loss

            batches.set_postfix_str(
                "Loss: {:.2f}, Running Loss: {:.2f}, LR: {:.4f}".format(
                    detached_loss, running_loss, lr
                )
            )

    return model, optimizer
