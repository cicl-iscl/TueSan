"""
Implements the training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from functools import partial
from torch.nn.utils import clip_grad_value_
from ray import tune as hyperparameter_tune
from torch.optim.lr_scheduler import OneCycleLR


def calculate_running_average(running, loss, gamma):
    if running is None:
        return loss
    else:
        return gamma * running + (1 - gamma) * loss


def train(
    model,
    optimizer,
    dataloader,
    epochs,
    device,
    tag_rules,
    max_lr,
    evaluate,
    tune,
    verbose=False,
):
    encoder = model["encoder"]
    stem_rule_classifier = model["stem_rule_classifier"]

    encoder.train()
    stem_rule_classifier.train()

    if not tag_rules:
        tag_classifier = model["tag_classifier"]
        tag_classifier.train()

    scheduler = OneCycleLR(
        optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(dataloader)
    )

    running_stem_loss = None
    running_tag_loss = None
    running_average = partial(calculate_running_average, gamma=0.95)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs):
        encoder.train()
        stem_rule_classifier.train()

        if verbose:
            batches = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        else:
            batches = dataloader

        for batch in batches:
            optimizer.zero_grad()
            # loss, (rule_accuracy, tag_accuracy), _ = batch_loss(batch)

            if tag_rules:
                tokens, stem_rules_true = batch
            else:
                tokens, stem_rules_true, tags_true = batch

            tokens = tokens.to(device)
            stem_rules_true = stem_rules_true.to(device)

            if not tag_rules:
                tags_true = tags_true.to(device)

            # Encode input sentence
            token_embeddings = encoder(tokens)
            y_pred_stem = stem_rule_classifier(token_embeddings)
            stem_loss = criterion(y_pred_stem, stem_rules_true)

            if not tag_rules:
                y_pred_tag = tag_classifier(token_embeddings)
                tag_loss = criterion(y_pred_tag, tags_true)
            else:
                tag_loss = torch.tensor(0.0)

            # Train model
            loss = stem_loss + tag_loss
            loss.backward()
            # Clip gradient values (can make training more stable)
            # clip_grad_value_(encoder.parameters(), 1.0)
            # clip_grad_value_(classifiers.parameters(), 10.0)
            optimizer.step()
            scheduler.step()

            # Display loss
            detached_stem_loss = stem_loss.detach().cpu().item()
            detached_tag_loss = tag_loss.detach().cpu().item()
            lr = scheduler.get_last_lr()[0]

            running_stem_loss = running_average(running_stem_loss, detached_stem_loss)
            running_tag_loss = running_average(running_tag_loss, detached_tag_loss)

            if verbose:
                batches.set_postfix_str(
                    "Stem Loss: {:.2f}, Tag Loss: {:.2f}, LR: {:.4f}".format(
                        running_stem_loss, running_tag_loss, lr
                    )
                )

        if tune and ((epoch + 1) % 5 == 0 or epochs < 5):
            with hyperparameter_tune.checkpoint_dir(epoch) as checkpoint_dir:
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                path = Path(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

            t2_score = evaluate(model)["task_2_tscore"]
            running_loss = running_stem_loss + running_tag_loss
            hyperparameter_tune.report(loss=running_loss, score=t2_score)

    if not tune:
        return model, optimizer
