"""
Implements the training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray import tune as hyperparameter_tune
from tqdm import tqdm
from pathlib import Path


def train(
    model,
    optimizer,
    criterion,
    dataloader,
    epochs,
    device,
    max_lr,
    evaluate,
    tune,
    verbose=False,
):
    model = model.to(device)

    # Use some fancy learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(dataloader)
    )

    # Mainting running average of loss
    running_loss = None

    # Train for given number of epochs
    for epoch in range(epochs):
        model.train()
        if verbose:
            batches = tqdm(dataloader, desc=f"Epoch: {epoch+1}")
        else:
            batches = dataloader

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

            if verbose:
                batches.set_postfix_str(
                    "Loss: {:.2f}, Running Loss: {:.2f}, LR: {:.4f}".format(
                        detached_loss, running_loss, lr
                    )
                )

        if tune:
            with hyperparameter_tune.checkpoint_dir(epoch) as checkpoint_dir:
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                path = Path(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

            # Evaluate
            t1_score = evaluate(model)["task_1_tscore"]
            hyperparameter_tune.report(loss=running_loss, score=t1_score)

    if not tune:
        return model, optimizer
