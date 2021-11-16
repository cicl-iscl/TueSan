"""
Implements the training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
import time


def train(model, optimizer, dataloader, epochs, device):
    sandhi_criterion = nn.CrossEntropyLoss(ignore_index=0)
    stem_criterion = nn.CTCLoss(zero_infinity=True)
    tag_criterion = nn.CTCLoss(zero_infinity=True)

    model = model.to(device)

    running_loss = None
    running_sandhi_loss = None
    running_stem_loss = None
    running_tag_loss = None
    gamma = 0.95

    for epoch in range(epochs):
        batches = tqdm(dataloader)

        for (
            source,
            sandhi_target,
            stem_target,
            tag_target,
            stem_target_lengths,
            tag_target_lengths,
        ) in batches:
            optimizer.zero_grad()

            source = source.to(device)
            source_lengths = (source != 0).sum(dim=-1).long().flatten()

            y_pred_sandhi, y_pred_stem, y_pred_tag = model(source)

            y_pred_sandhi = y_pred_sandhi.flatten(end_dim=-2)
            sandhi_target = sandhi_target.flatten().long().to(device)
            sandhi_loss = sandhi_criterion(y_pred_sandhi, sandhi_target)

            y_pred_stem = y_pred_stem.transpose(0, 1)
            y_pred_stem = F.log_softmax(y_pred_stem, dim=-1)
            stem_loss = stem_criterion(
                y_pred_stem, stem_target, source_lengths, stem_target_lengths
            )

            y_pred_tag = y_pred_tag.transpose(0, 1)
            y_pred_tag = F.log_softmax(y_pred_tag, dim=-1)
            tag_loss = tag_criterion(
                y_pred_tag, tag_target, source_lengths, tag_target_lengths
            )

            loss = sandhi_loss + stem_loss + tag_loss

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            print(stem_target)
            print(stem_target_lengths)
            print()
            print(tag_target)
            print(tag_target_lengths)
            print()
            print(source_lengths)

            if torch.isnan(loss):
                raise ValueError

            detached_loss = loss.detach().cpu().item()
            detached_sandhi_loss = sandhi_loss.detach().cpu().item()
            detached_stem_loss = stem_loss.detach().cpu().item()
            detached_tag_loss = tag_loss.detach().cpu().item()

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
                "Loss: {:.2f}, Sandhi Loss: {:.2f}, Stem Loss: {:.2f}, Tag Loss: {:.2f}".format(
                    running_loss,
                    running_sandhi_loss,
                    running_stem_loss,
                    running_tag_loss,
                )
            )
            time.sleep(0.2)

    return model, optimizer
