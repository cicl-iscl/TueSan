"""Generate training data for Task 1
    with or without extra translit.

    For alignment we need to also convert the tokens.
"""

from tqdm import tqdm
from helpers import get_task1_IO
from edit_distance import SequenceMatcher as align


KEEP_RULE = "<COPY>"
APPEND_SPACE_RULE = "<INSERT SPACE>"


def rule_normalise(rule):
    premise, consequence = rule

    # Only have 1 rule for all if we just copy the char
    if premise == consequence:
        return KEEP_RULE

    # Only have 1 rule for all if we only insert 1 space
    # after the char
    elif premise + " " == consequence:
        return APPEND_SPACE_RULE

    else:
        return consequence


def sent2rules(source, target):
    # Source and target are strings
    rules = []

    # Find alignment between source and target chars
    alignment = align(a=source, b=target)
    alignment = alignment.get_matching_blocks()
    s_indices, t_indices, _ = zip(*alignment)

    # Take extra care of the first chunk:
    # If the first char of source is not aligned to anything, we have to include it manually
    s_idx, t_idx = s_indices[0], t_indices[0]
    if s_idx != 0:
        rule = (source[:s_idx], target[:t_idx])
        rules.append(rule)

    # Introduce rules that map contiguous aligned source and target chunks
    for k in range(len(s_indices)):
        s_start_idx = s_indices[k]
        s_stop_idx = s_indices[k + 1] if k + 1 < len(s_indices) else None
        t_start_idx = t_indices[k]
        t_stop_idx = t_indices[k + 1] if k + 1 < len(s_indices) else None

        # If we map 1 source char to something, don't do anything special
        if s_stop_idx is None or s_stop_idx - s_start_idx == 1:
            premise = source[s_start_idx:s_stop_idx]  # Rule input
            consequence = target[t_start_idx:t_stop_idx]  # Rule output

            rule = (premise, consequence)
            rules.append(rule)

        # If we map multiple source chars, do
        #  1. Map first source char of chunk to aligned target char
        #  2. Map following source chars to following aligned target chars
        else:
            rules.append((source[s_start_idx], target[t_start_idx]))
            rules.append(
                (
                    source[s_start_idx + 1 : s_stop_idx],
                    target[t_start_idx + 1 : t_stop_idx],
                )
            )

    # Reconstruct target:
    # Sanity check if rule construction is ok
    reconstructed_sequence = []
    for rule in rules:
        reconstructed_sequence += rule[1]
    reconstructed_sequence = "".join(reconstructed_sequence)

    assert reconstructed_sequence == target

    # Check whether rules are sensitive:
    for premise, consequence in rules:
        assert len(premise) <= 5 and len(consequence) <= 5

    # Generate input and output sequences
    input = []
    output = []

    for rule in rules:
        premise, consequence = rule
        rule = rule_normalise(rule)

        for char in premise:
            input.append(char)
            output.append(rule)

    return input, output


def construct_train_dataset(data):
    dataset = []
    rules = set()
    discarded = 0

    # Extract rule for each sentence in dataset
    for sandhied, unsandhied_tokenized in tqdm(data):
        unsandhied = " ".join(unsandhied_tokenized)

        try:
            # Get input and output sequences of equal length
            input, target = sent2rules(sandhied, unsandhied)
            rules.update(set(target))
            dataset.append((input, target))

        # Malformed sentences are discarded
        except AssertionError:
            discarded += 1

    rules = list(sorted(rules))
    return dataset, rules, discarded


def construct_eval_dataset(data):
    return [(sandhied, " ".join(unsandhied)) for sandhied, unsandhied in tqdm(data)]
