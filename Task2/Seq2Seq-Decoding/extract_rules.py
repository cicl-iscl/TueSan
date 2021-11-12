"""
Implements extraction of prefix/postfix rules
"""

import numpy as np

from tqdm import tqdm
from collections import defaultdict

from uni2intern import unicode_to_internal_transliteration as to_intern


UNK_RULE = "<Other>"


def get_token_stem_pairs(data):
    token_stem_pairs = []

    for sentence, labels in data:
        tokens = sentence.split()
        stems, tags = zip(*labels)

        if len(tokens) != len(stems):
            continue

        token_stem_pairs.extend(list(zip(tokens, stems, tags)))

    token_stem_pairs = list(sorted(set(token_stem_pairs)))
    return token_stem_pairs


def get_length_of_overlap(token, stem, start_index):
    """
    Gets the maximum number k of chars in token starting at start_index
    that are equal to the first k chars in stem
    """
    length = 0
    for i in range(0, min(len(stem), len(token) - start_index)):
        if stem[i] == token[start_index + i]:
            length += 1
        else:
            break

    return length


def get_rule(token, stem):
    # Finds all indices of chars in token that are equal to
    # first char in stem
    start_indices = np.nonzero([char == stem[0] for char in token])[0]

    # If no overlap, no rule
    if len(start_indices) == 0:
        return None

    # Find length of overlapping char segments
    matching_segment_lengths = [
        get_length_of_overlap(token, stem, start_index)
        for start_index in start_indices
    ]

    # Take longest overlapping char segment
    # as 'root' (may be different from linguistic root)
    best_idx = np.argmax(matching_segment_lengths)
    best_length = matching_segment_lengths[best_idx]
    best_idx = start_indices[best_idx]

    # Prefix = part before 'root'
    prefix = token[:best_idx]
    # Suffix = part after 'root'
    suffix = token[best_idx + best_length :]
    # Replaced suffix
    stem_suffix = stem[best_length:]

    return prefix, suffix, stem_suffix


def get_rules(data, use_tag=True):
    # Make pairs (token, stem, tag)
    token_stem_pairs = get_token_stem_pairs(data)

    # Extract rule from each (token, stem) pair, if possible
    rules = defaultdict(int)
    for token, stem, tag in token_stem_pairs:
        rule = get_rule(token, stem)

        if rule is None:
            continue

        if use_tag:
            rule = (*rule, tag)

        rules[rule] += 1

    return rules


def rule_is_applicable(rule, token, apply_rule=True):
    candidate_stem = token[:]

    prefix, suffix, stem_suffix = rule[:3]
    applicable = False

    if candidate_stem.startswith(prefix):
        candidate_stem = candidate_stem[len(prefix) :]

        if candidate_stem.endswith(suffix):
            candidate_stem = candidate_stem[
                : len(candidate_stem) - len(suffix)
            ]
            candidate_stem += stem_suffix
            applicable = True

    candidate_stem = None if not applicable else candidate_stem

    if apply_rule:
        return applicable, candidate_stem
    else:
        return applicable


def get_applicable_rules(token, stem, tag, rules, check_tag=True):
    applicable_rules = []

    for rule in rules:
        if check_tag and rule[3] != tag:
            continue

        applicable, candidate_stem = rule_is_applicable(rule, token)
        if candidate_stem == stem:
            applicable_rules.append(rule)

    return applicable_rules


def get_token_rule_mapping(data, rules, use_tag=True, eval=False):

    token_rule_mapping = []

    for sentence, labels in tqdm(data):
        sentence = sentence.split()
        stems, tags = zip(*labels)

        if not eval and len(sentence) != len(stems):
            continue

        sentence_mapping = []

        for token, stem, tag in zip(sentence, stems, tags):
            applicable_rules = get_applicable_rules(
                token, stem, tag, rules, check_tag=use_tag
            )

            if len(applicable_rules) == 0:
                sentence_mapping.append((token, stem, tag, UNK_RULE))
            elif len(applicable_rules) == 1:
                applicable_rule = applicable_rules[0]
                sentence_mapping.append((token, stem, tag, applicable_rule))
            else:
                applicable_rule = np.random.choice(applicable_rules)
                sentence_mapping.append((token, stem, tag, applicable_rule))

        token_rule_mapping.append(sentence_mapping)

    return token_rule_mapping


def generate_stems_from_rules(token, rules):
    candidate_stems, candidate_tags = set(), set()

    for rule in rules:
        applicable, candidate_stem = rule_is_applicable(rule, token)
        if applicable:
            candidate_stems.add(candidate_stem)
            candidate_tags.add(rule[3])

    candidate_stems = list(sorted(candidate_stems))
    candidate_tags = list(sorted(candidate_tags))

    return candidate_stems, candidate_tags
