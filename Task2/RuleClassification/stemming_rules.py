import numpy as np
from tqdm import tqdm

UNK_RULE = "<OTHER>"  # Unknown rule


def matching_prefix_length(s1, s2):
    length = 0

    for char1, char2 in zip(s1, s2):
        if char1 != char2:
            break
        else:
            length += 1

    return length


def align_stem(token, stem):
    table_shape = (len(token), len(stem))
    prefix_match_lengths = np.zeros(table_shape)

    for i in range(len(token)):
        for j in range(len(stem)):
            prefix_match_lengths[i, j] = matching_prefix_length(token[i:], stem[j:])

    best_index = np.argmax(prefix_match_lengths, axis=None)
    best_index = np.unravel_index(best_index, table_shape)

    token_start, stem_start = best_index
    length = prefix_match_lengths[token_start, stem_start]
    length = int(length)

    return token_start, stem_start, length


def token2stemmingrule(token, stem, tag, tag_rules):
    token, stem = token.strip(), stem.strip()

    # Align stem with token to extract prefix and suffix
    token_start, stem_start, length = align_stem(token, stem)

    token_prefix = token[:token_start]
    token_suffix = token[token_start + length :]

    stem_prefix = stem[:stem_start]
    stem_suffix = stem[stem_start + length :]

    if tag_rules:
        rule = ((token_prefix, token_suffix), (stem_prefix, stem_suffix), tag)
    else:
        rule = ((token_prefix, token_suffix), (stem_prefix, stem_suffix))

    return rule


def apply_rule(token, rule):
    (token_prefix, token_suffix), (stem_prefix, stem_suffix), *_ = rule
    if not (token.startswith(token_prefix) and token.endswith(token_suffix)):
        raise ValueError(
            f"({token_prefix}, {token_suffix}) -> ({stem_prefix}, {stem_suffix}) is invalid for {token}"
        )

    if len(token_prefix) > 0:
        token = token[len(token_prefix) :]

    if len(token_suffix) > 0:
        token = token[: -len(token_suffix)]

    return stem_prefix + token + stem_suffix


def evaluate_coverage(eval_data, rules, logger, tag_rules):
    logger.debug("Evaluating stem rule coverage")
    covered, covered_unique = 0, 0
    not_covered, not_covered_unique = 0, 0
    seen_tokens = set()
    stem_candidate_lengths = []

    for tokens, labels in tqdm(eval_data):
        tokens = tokens.split()

        for token, (stem, tag) in zip(tokens, labels):
            stem_candidates = set()
            tag_candidates = set()
            for rule in rules:
                try:
                    stem_candidate = apply_rule(token, rule)
                    stem_candidates.add(stem_candidate)

                    if tag_rules:
                        tag_candidates.add(rule[-1])

                except ValueError:
                    continue

            if stem in stem_candidates and (tag in tag_candidates or not tag_rules):
                covered += 1
                if token not in seen_tokens:
                    covered_unique += 1

            else:
                not_covered += 1
                if token not in seen_tokens:
                    not_covered_unique += 1

            seen_tokens.add(token)
            stem_candidate_lengths.append(len(stem_candidates))

    covered_percentage = covered / (covered + not_covered)
    covered_percentage *= 100

    covered_unique_percentage = covered_unique / (covered_unique + not_covered_unique)
    covered_unique_percentage *= 100

    logger.debug(
        "Collected rules cover {:.2f}% of eval stems (all)".format(covered_percentage)
    )
    logger.debug(
        "Collected rules cover {:.2f}% of eval stems (unique tokens)".format(
            covered_unique_percentage
        )
    )

    logger.debug(f"Avg. # candidate stems {np.mean(stem_candidate_lengths)}")

    return covered / (covered + not_covered)
