"""
Implements the training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
import time


def train(model, optimizer, dataloader, epochs, device):
    sandhi_criterion = nn.CrossEntropyLoss(ignore_index=0)
    stem_criterion = nn.CrossEntropyLoss(ignore_index=0)
    tag_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    scheduler = OneCycleLR(optimizer, 0.1, epochs = epochs, steps_per_epoch = len(dataloader))

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
        batches = tqdm(dataloader, desc = f"Epoch {epoch}")

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

            loss.backward()
            # clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if torch.isnan(loss):
                raise ValueError

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

            batches.set_postfix_str(
                "Loss: {:.2f}, Sandhi Loss: {:.2f}, Stem Loss: {:.2f}, Tag Loss: {:.2f}, LR: {:.4f}".format(
                    running_loss,
                    running_sandhi_loss,
                    running_stem_loss,
                    running_tag_loss,
                    lr
                )
            )
            # time.sleep(0.2)

    return model, optimizer
