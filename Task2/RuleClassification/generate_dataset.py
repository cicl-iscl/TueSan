"""
Generate training data for Task 1
with or without extra translit.

For alignment we need to also convert the tokens.
"""

from tqdm import tqdm
from collections import defaultdict

from stemming_rules import token2stemmingrule
from stemming_rules import UNK_RULE


def construct_train_dataset(data, tag_rules, stemming_rule_cutoff):
    dataset = []
    stemming_rules = dict()
    stemming_rules_count = defaultdict(int)

    discarded = 0

    for segmented, labels in data:
        tokens = segmented.split()

        try:
            assert len(tokens) == len(labels)
        except AssertionError:
            discarded += 1
            continue

        # Generate stemming rules
        stem_target = []
        tag_target = []

        for token, (stem, tag) in zip(tokens, labels):
            key = (token, stem) if not tag_rules else (token, stem, tag)
            try:
                stemming_rule = stemming_rules[key]
                stemming_rules_count[stemming_rule] += 1
            except KeyError:
                stemming_rule = token2stemmingrule(token, stem, tag, tag_rules)
                stemming_rules[key] = stemming_rule
                stemming_rules_count[stemming_rule] += 1

            stem_target.append(stemming_rule)
            tag_target.append(tag)

        dataset.append([tokens, stem_target, tag_target])

    # Clean stem rules (only use if freq. > cutoff)
    cleaned_dataset = []
    cleaned_stemming_rules = set()
    all_tags = set()

    for tokens, stem_target, tag_target in dataset:
        cleaned_stem_target = []

        for rule, tag in zip(stem_target, tag_target):
            if stemming_rules_count[rule] > stemming_rule_cutoff:
                cleaned_stem_target.append(rule)
                cleaned_stemming_rules.add(rule)
                if tag_rules:
                    all_tags.add(rule[-1])
                else:
                    all_tags.add(tag)
            else:
                cleaned_stem_target.append(UNK_RULE)
                if not tag_rules:
                    all_tags.add(tag)

        cleaned = (tokens, cleaned_stem_target, tag_target)
        cleaned_dataset.append(cleaned)

    return cleaned_dataset, cleaned_stemming_rules, all_tags, discarded
