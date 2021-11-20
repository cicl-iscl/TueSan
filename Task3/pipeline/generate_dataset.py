"""
Generate training data for Task 1
with or without extra translit.

For alignment we need to also convert the tokens.
"""

from tqdm import tqdm
from collections import defaultdict

from sandhi_rules import sent2sandhirules
from sandhi_rules import KEEP_RULE, APPEND_SPACE_RULE
from stemming_rules import token2stemmingrule
from stemming_rules import UNK_RULE


def get_boundaries(sandhied, sandhi_rules):
    boundaries = []
    start = 0
    previous_rule = None

    for i, (char, rule) in enumerate(zip(sandhied, sandhi_rules)):
        if rule == APPEND_SPACE_RULE:
            boundaries.append((start, i + 1))
            start = i + 1

        elif char == " ":
            boundaries.append((start, i))
            start = i + 1

        elif rule != KEEP_RULE and " " in rule and rule != previous_rule:
            for _ in range(list(rule).count(" ")):
                boundaries.append((start, i + 1))
            start = i

        previous_rule = rule

    boundaries.append((start, len(sandhied)))
    return boundaries


def construct_train_dataset(data):
    dataset = []
    sandhi_rules = defaultdict(int)
    stemming_rules = dict()
    stemming_rules_count = defaultdict(int)
    all_tags = set()

    discarded = 0
    stemming_rule_cutoff = 10

    for sandhied, labels in tqdm(data):
        unsandhied_tokenized, stems, tags = zip(*labels)

        # Extract sandhi rules for each sentence in dataset
        try:
            unsandhied = " ".join(unsandhied_tokenized)
            # Get input and output sequences of equal length
            _, sandhi_target = sent2sandhirules(sandhied, unsandhied)

            boundaries = get_boundaries(sandhied, sandhi_target)
            for sandhi_rule in sandhi_target:
                sandhi_rules[sandhi_rule] += 1

            # try:
            assert len(boundaries) == len(unsandhied_tokenized)
            # except AssertionError:
            #    print(unsandhied)
            #    print(sandhied)
            #    print(boundaries)
            #    print([sandhied[i:j] for i, j in boundaries])
            #    print()
            #    raise

        # Malformed sentences are discarded
        except AssertionError:
            discarded += 1
            continue

        # Generate stemming rules
        stem_target = []

        for token, stem in zip(unsandhied_tokenized, stems):
            try:
                stemming_rule = stemming_rules[(token, stem)]
                stemming_rules_count[stemming_rule] += 1
            except KeyError:
                stemming_rule = token2stemmingrule(token, stem)
                stemming_rules[(token, stem)] = stemming_rule
                stemming_rules_count[stemming_rule] += 1

            stem_target.append(stemming_rule)

        # Tags stay unchanged
        tag_target = list(tags)
        all_tags.update(set(tags))

        dataset.append([sandhied, sandhi_target, stem_target, tag_target, boundaries])

    # Clean stem rules (only use if freq. > cutoff)
    cleaned_dataset = []
    cleaned_stemming_rules = set()

    for sandhied, sandhi_target, stem_target, tag_target, boundaries in dataset:
        cleaned_stem_target = []

        for rule in stem_target:
            if stemming_rules_count[rule] > stemming_rule_cutoff:
                cleaned_stem_target.append(rule)
                cleaned_stemming_rules.add(rule)
            else:
                cleaned_stem_target.append(UNK_RULE)

        cleaned = (sandhied, sandhi_target, cleaned_stem_target, tag_target, boundaries)
        cleaned_dataset.append(cleaned)

    return cleaned_dataset, sandhi_rules, cleaned_stemming_rules, all_tags, discarded
