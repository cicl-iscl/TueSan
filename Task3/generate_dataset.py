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
    stemming_rules = defaultdict(int)

    discarded = 0
    stemming_rule_cutoff = 5

    for sandhied, labels in tqdm(data):
        unsandhied_tokenized, stems, tags = zip(*labels)

        # Extract sandhi rules for each sentence in dataset
        try:
            unsandhied = " ".join(unsandhied_tokenized)
            # Get input and output sequences of equal length
            _, sandhi_target = sent2rules(sandhied, unsandhied)
            for sandhi_rule in target:
                sandhi_rules[sandhi_rule] += 1

        # Malformed sentences are discarded
        except AssertionError:
            discarded += 1
            continue

        # Generate stemming rules
        stem_target = []

        for token, stem in zip(unsandhied_tokenized, stems):
            stemming_rule = token2stemmingrule(token, stem)
            stemming_rules[stemming_rule] += 1

            stem_target.append(stemming_rule)

        # Tags stay unchanged
        tag_target = list(tags)

        dataset.append([sandhied, sandhi_target, stem_target, tag_target])

    # Clean stem rules (only use if freq. > cutoff)
    cleaned_dataset = []
    for sandhied, sandhi_target, stem_target, tag_target in dataset:
        cleaned_rules = []

        for rule in stem_target:
            if stemming_rules[rule] > stemming_rule_cutoff:
                cleaned_rules.append(rule)
            else:
                cleaned_rules.append(UNK_RULE)

        cleaned = (sandhi, sandhi_target, cleaned_rules, tag_target)
        cleaned_dataset.append(cleaned)

    return cleaned_dataset, sandhi_rules, stemming_rules, discarded
