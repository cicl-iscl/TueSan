"""Generate training data for Task 1
    with or without extra translit.

    For alignment we need to also convert the tokens.
"""

from tqdm import tqdm
from collections import defaultdict

from sandhi_rules import sent2sandhirules
from stemming_rules import token2stemmingrule
from stemming_rules import UNK_RULE


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
            for sandhi_rule in sandhi_target:
                sandhi_rules[sandhi_rule] += 1

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

        dataset.append([sandhied, sandhi_target, stem_target, tag_target])

    # Clean stem rules (only use if freq. > cutoff)
    cleaned_dataset = []
    cleaned_stemming_rules = set()

    for sandhied, sandhi_target, stem_target, tag_target in dataset:
        cleaned_stem_target = []

        for rule in stem_target:
            if stemming_rules_count[rule] > stemming_rule_cutoff:
                cleaned_stem_target.append(rule)
                cleaned_stemming_rules.add(rule)
            else:
                cleaned_stem_target.append(UNK_RULE)

        cleaned = (sandhied, sandhi_target, cleaned_stem_target, tag_target)
        cleaned_dataset.append(cleaned)

    return cleaned_dataset, sandhi_rules, cleaned_stemming_rules, all_tags, discarded
