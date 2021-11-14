"""
Implements the training loop
"""

import torch
import torch.nn.functional as F

from tqdm import tqdm
from functools import partial
from loss import masked_cross_entropy_loss
from torch.nn.utils import clip_grad_value_


def get_loss(batch, encoder, decoder, tag_classifier, device, pad_tag):
    inputs, stems, tags = batch
    stems, tags = stems.long(), tags.long()

    inputs = inputs.to(device)
    stems = stems.to(device)
    tags = tags.to(device)

    # Encode input sentence
    encoded, char_embeddings, token_lengths = encoder(inputs)

    # Predict morphological tags
    y_pred_tags = tag_classifier(encoded)

    # Predict stems (teacher forcing)
    decoder_indices = token_lengths.nonzero().flatten()
    stems = stems.flatten(end_dim=-2)
    stems = stems[decoder_indices].contiguous()

    y_pred_decoder = decoder(stems[:, :-1], encoded, char_embeddings, token_lengths)

    # Calculate losses
    tag_indices = (tags != pad_tag).flatten().nonzero().flatten()
    tags = tags.flatten()[tag_indices].contiguous()
    tag_loss = F.cross_entropy(y_pred_tags, tags)

    stems = stems[:, 1:].flatten()
    stem_indices = stems.nonzero().flatten()
    stems = stems[stem_indices].contiguous()

    y_pred_decoder = y_pred_decoder.flatten(end_dim=-2)
    y_pred_decoder = y_pred_decoder[stem_indices].contiguous()

    decoder_loss = F.cross_entropy(y_pred_decoder, stems)

    return tag_loss, decoder_loss


def train(model, optimizer, dataloader, epochs, device, pad_tag):
    encoder = model["encoder"]
    decoder = model["decoder"]
    tag_classifier = model["tag_classifier"]

    encoder.train()
    decoder.train()
    tag_classifier.train()

    batch_loss = partial(
        get_loss,
        encoder=encoder,
        decoder=decoder,
        tag_classifier=tag_classifier,
        device=device,
        pad_tag=pad_tag,
    )
    running_loss = None

    for epoch in range(epochs):
        batches = tqdm(dataloader)

        for batch in batches:
            optimizer.zero_grad()
            tag_loss, decoder_loss = batch_loss(batch)

            # Train model
            loss = tag_loss + decoder_loss
            loss.backward()
            # Clip gradient values (can make training more stable)
            clip_grad_value_(encoder.parameters(), 1.0)
            clip_grad_value_(decoder.parameters(), 1.0)
            clip_grad_value_(tag_classifier.parameters(), 10.0)
            optimizer.step()

            # Display loss
            detached_loss = loss.detach().cpu().item()
            detached_tag_loss = tag_loss.detach().cpu().item()
            detached_decoder_loss = decoder_loss.detach().cpu().item()

            if running_loss is None:
                running_loss = detached_loss
            else:
                running_loss = 0.99 * running_loss + 0.01 * detached_loss

            batches.set_postfix_str(
                "Total Running Loss: {:.2f}, Tag Loss: {:.2f}, Decoder Loss: {:.2f}".format(
                    running_loss, detached_tag_loss, detached_decoder_loss
                )
            )

    return model, optimizer
