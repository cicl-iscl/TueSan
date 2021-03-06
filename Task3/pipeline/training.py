"""
Implements the training loop
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_

from ray import tune as hyperparameter_tune

from pathlib import Path
from tqdm import tqdm
import time


def train(
    model, optimizer, dataloader, epochs, device, max_lr, evaluate, tune, verbose=False,
):
    sandhi_criterion = nn.CrossEntropyLoss(ignore_index=0)
    stem_criterion = nn.CrossEntropyLoss(ignore_index=0)
    tag_criterion = nn.CrossEntropyLoss(ignore_index=0)

    scheduler = OneCycleLR(
        optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(dataloader)
    )

    model = model.to(device)
    segmenter = model["segmenter"]
    classifier = model["classifier"]

    segmenter.train()
    classifier.train()

    running_loss = None
    running_sandhi_loss = None
    running_stem_loss = None
    running_tag_loss = None
    gamma = 0.95

    for epoch in range(epochs):
        if verbose:
            batches = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        else:
            batches = dataloader
        
        segmenter.train()
        classifier.train()

        for (source, sandhi_target, stem_target, tag_target, boundaries) in batches:
            optimizer.zero_grad()

            sandhi_target = sandhi_target.to(device)
            stem_target = stem_target.to(device)
            tag_target = tag_target.to(device)

            source = source.to(device)
            # source_lengths = (source != 0).sum(dim=-1).long().flatten()

            y_pred_sandhi, char_embeddings = segmenter(source)

            y_pred_sandhi = y_pred_sandhi.flatten(end_dim=-2)
            sandhi_target = sandhi_target.flatten().long().to(device)
            sandhi_loss = sandhi_criterion(y_pred_sandhi, sandhi_target)

            y_pred_stem, y_pred_tag = classifier(char_embeddings, boundaries)
            stem_loss = stem_criterion(y_pred_stem, stem_target)
            tag_loss = tag_criterion(y_pred_tag, tag_target)
            loss = sandhi_loss + stem_loss + tag_loss
            
            if torch.isnan(loss):
                continue

            loss.backward()
            # clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            detached_loss = loss.detach().cpu().item()
            detached_sandhi_loss = sandhi_loss.detach().cpu().item()
            detached_stem_loss = stem_loss.detach().cpu().item()
            detached_tag_loss = tag_loss.detach().cpu().item()
            lr = scheduler.get_last_lr()[0]

            if running_loss is None:
                running_loss = detached_loss
                running_sandhi_loss = detached_sandhi_loss
                running_stem_loss = detached_stem_loss
                running_tag_loss = detached_tag_loss

            else:
                running_loss = gamma * running_loss + (1 - gamma) * detached_loss
                running_sandhi_loss = (
                    gamma * running_sandhi_loss + (1 - gamma) * detached_sandhi_loss
                )
                running_stem_loss = (
                    gamma * running_stem_loss + (1 - gamma) * detached_stem_loss
                )
                running_tag_loss = (
                    gamma * running_tag_loss + (1 - gamma) * detached_tag_loss
                )

            if verbose:
                batches.set_postfix_str(
                    "Loss: {:.2f}, Sandhi Loss: {:.2f}, Stem Loss: {:.2f}, Tag Loss: {:.2f}, LR: {:.4f}".format(
                        running_loss,
                        running_sandhi_loss,
                        running_stem_loss,
                        running_tag_loss,
                        lr,
                    )
                )
        # Evaluate every 5 epochs
        if tune and ((epoch + 1) % 5 == 0 or epochs < 5):
            # os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
            with hyperparameter_tune.checkpoint_dir(epoch) as checkpoint_dir:
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                path = Path(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

            t3_score = evaluate(model)["task_3_tscore"]
            hyperparameter_tune.report(loss=running_loss, score=t3_score)

    if not tune:
        return model, optimizer
