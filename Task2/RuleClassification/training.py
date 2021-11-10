"""
Implements the training loop
"""

import torch
import torch.nn.functional as F

from tqdm import tqdm
from functools import partial
from loss import masked_cross_entropy_loss
from torch.nn.utils import clip_grad_value_

from sklearn.metrics import accuracy_score


def classifier_loss(classifier, encoded, y_true):
    # Predict
    y_pred = classifier(encoded)
    y_pred = y_pred.flatten(end_dim=-2)
    y_true = y_true.flatten()

    mask = torch.nonzero((y_true != 0)).flatten()
    y_true = y_true[mask].contiguous()
    loss = F.cross_entropy(y_pred, y_true)

    with torch.no_grad():
        y_pred_max = torch.argmax(y_pred, dim=-1).cpu().numpy()
        y_true = y_true.cpu().numpy()

        accuracy = accuracy_score(y_pred_max, y_true)

    return loss, accuracy, y_pred


def get_loss(batch, encoder, classifiers, device):
    tokens = batch[0].to(device)

    # Encode input sentence
    token_embeddings = encoder(tokens)

    # Predict rules
    rules_true = batch[1].long().to(device)
    rule_loss, rule_accuracy, y_pred_rule = classifier_loss(
        classifiers["rules"], token_embeddings, rules_true
    )

    # Predict tags and calculate loss (optional)
    if len(batch) == 3:
        tags_true = batch[2].long().to(device)
        tag_loss, tag_accuracy, y_pred_tag = classifier_loss(
            classifiers["tags"], token_embeddings, tags_true
        )
    else:
        tag_loss = 0.0
        tag_accuracy = rule_accuracy
        y_pred_tag = None

    accuracy = (rule_accuracy + tag_accuracy) / 2

    return rule_loss + tag_loss, accuracy, (y_pred_rule, y_pred_tag)


def train(model, optimizer, dataloader, epochs, device):
    encoder = model["encoder"]
    classifiers = model["classifiers"]

    encoder.train()
    classifiers.train()

    batch_loss = partial(
        get_loss, encoder=encoder, classifiers=classifiers, device=device
    )
    running_loss = None
    running_accuracy = 0.0

    for epoch in range(epochs):
        batches = tqdm(dataloader)

        for batch in batches:
            optimizer.zero_grad()
            loss, accuracy, _ = batch_loss(batch)

            # Train model
            loss.backward()
            # Clip gradient values (can make training more stable)
            clip_grad_value_(encoder.parameters(), 1.0)
            clip_grad_value_(classifiers.parameters(), 10.0)
            optimizer.step()

            # Display loss
            detached_loss = loss.detach().cpu().item()

            if running_loss is None:
                running_loss = detached_loss
            else:
                running_loss = 0.99 * running_loss + 0.01 * detached_loss

            running_accuracy = 0.95 * running_accuracy + 0.05 * accuracy

            batches.set_postfix_str(
                "Total Running Loss: {:.2f}, Running Acc.: {:.2f}, Batch Loss: {:.2f}".format(
                    running_loss, running_accuracy, detached_loss
                )
            )

    return model, optimizer
