"""
'Decoding' is the process of incrementally collecting predictions from
the decoder (characters in our case). We implement 2 types of decoding:

Greedy decoding simply takes the character with highest prediction score
for each timestep.

Informed decoding has access to a (potentially empty) list of possible
stems and thereby only chooses from the valid characters. Among the
valid characters, decoding again chooses the character with highest prediction
score. Decoding also keeps track of which possible stems become invalid
during decoding, because they are inconsistent with some predicted
character.
"""

import torch

from vocabulary import SOS_TOKEN, EOS_TOKEN


# Inspired by https://github.com/joeynmt/joeynmt/blob/master/joeynmt/search.py


def safe_max(vals):
    """
    Returns 0 for empty sequences instead of raising an error.
    """
    if len(vals) == 0:
        return 0
    return max(vals)


# Informed decoding constrains the stems that can be predicted
# to a given list of possible stems
def informed_decoding(
    decoder,
    encoder_output,
    char_embeddings,
    token_lengths,
    allowed_stems,
    char2index,
    index2char,
):
    # Decoder: Trained neural network for predicting stems
    # encoder_output: Embedded input sentence
    #                 Embedding for each input token
    # allowed_stems:  For each token, a (possibly empty) list of possible stems
    #                 Taken from a dictionary
    #
    # Flatten encoder output: Each batch element corresponds to exactly 1 token
    decoder_input = encoder_output.flatten(end_dim=-2)
    batch_size = decoder_input.shape[0]
    # We have to make sure that tokens and allowed stems are
    # correctly aligned
    assert len(allowed_stems) == batch_size

    # Maximum length if length of longest allowed stem in batch
    max_len = safe_max(
        [safe_max([len(stem) for stem in stems]) for stems in allowed_stems]
    )
    # Start with start of sequence tokens
    sos_tokens = torch.LongTensor([char2index[SOS_TOKEN]] * batch_size)
    sos_tokens = sos_tokens.to(encoder_output.device)

    last_output = sos_tokens
    hidden = None

    predicted_indices = []
    # For each timestep, predict characters
    for step in range(max_len):
        # First, calculate prediction probabilities
        output, hidden = decoder._forward_step(
            last_output, decoder_input, char_embeddings, token_lengths, hidden=hidden
        )
        output = output.detach().cpu()

        # Now, we process each token alone
        current_indices = []
        for i, prediction_probs in enumerate(output):
            # Get allowed stems
            current_allowed_stems = allowed_stems[i]
            # If there are no options, we have no information
            # so we just take the character with highest prediction
            # probability
            if len(current_allowed_stems) == 0:
                predicted_index = torch.argmax(prediction_probs).item()

            # If we have only 1 possibility, we just take the corresponding
            # character
            elif len(current_allowed_stems) == 1:
                allowed_stem = current_allowed_stems[0]
                # If we have already finished this token
                # (current time step is higher than length of only possibility)
                # we predict end-of-sequence
                if step >= len(allowed_stem):
                    predicted_index = char2index[EOS_TOKEN]
                # Else, predict character at current timestep
                else:
                    predicted_index = char2index[allowed_stem[step]]

            else:
                # Collect possible chars
                allowed_indices = []
                # For each stem, look for char at current timestep
                for allowed_stem in current_allowed_stems:
                    # If stem too short, skip
                    if step < len(allowed_stem):
                        current_char = allowed_stem[step]
                        # If char unknown, skip
                        if current_char in char2index:
                            allowed_indices.append(char2index[current_char])
                # Deduplicate chars
                allowed_indices = list(set(allowed_indices))

                # If we don't have any allowed stems left
                # assume we're done (predict end of sequence)
                if len(allowed_indices) == 0:
                    predicted_index = char2index[EOS_TOKEN]
                # Else, predict char with highest prediction probability
                # according to decoder
                else:
                    predicted_index = torch.argmax(prediction_probs[allowed_indices])
                    predicted_index = allowed_indices[predicted_index]

                    # Remove stems that don't agree with actually predicted
                    # character
                    predicted_char = index2char[predicted_index]
                    updated_allowed_stems = [
                        stem
                        for stem in current_allowed_stems
                        if len(stem) > step and stem[step] == predicted_char
                    ]
                    allowed_stems[i] = updated_allowed_stems

            current_indices.append(predicted_index)

        current_indices = torch.LongTensor(current_indices)
        predicted_indices.append(current_indices)
        last_output = current_indices.to(encoder_output.device)

    predicted_indices = torch.stack(predicted_indices).transpose(0, 1)
    return predicted_indices


def greedy_decoding(
    decoder, encoder_output, char_embeddings, token_lengths, char2index, max_len=20
):
    eos_index = char2index[EOS_TOKEN]
    # Flatten encoder output: Each batch element corresponds to exactly 1 token
    decoder_input = encoder_output.reshape(-1, encoder_output.shape[-1])

    # Start with start of sequence tokens
    batch_size = decoder_input.shape[0]
    sos_tokens = torch.LongTensor([char2index[SOS_TOKEN]] * batch_size)
    sos_tokens = sos_tokens.to(encoder_output.device)

    last_output = sos_tokens
    hidden = None

    # Record which stems are done
    finished = encoder_output.new_zeros(batch_size).bool()
    predicted_indices = []

    # For each timestep, predict characters and greedily take most likely
    # next character according to decoder
    for _ in range(max_len):
        output, hidden = decoder._forward_step(
            last_output, decoder_input, char_embeddings, token_lengths, hidden=hidden
        )
        output = torch.argmax(output, dim=-1)
        predicted_indices.append(output.detach().cpu())

        last_output = output

        finished += output == eos_index
        if torch.all(finished >= 1):
            break

    predicted_indices = torch.stack(predicted_indices).transpose(0, 1)
    return predicted_indices
