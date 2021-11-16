import numpy as np

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
    prefix_match_lengths = np.zeros(*table_shape)
    
    for i in range(len(token)):
        for j in range(len(stem)):
            prefix_match_lengths[i, j] = matching_prefix_length(token[i:], stem[j:])
    
    best_index = np.argmax(prefix_match_lengths, axis=None)
    best_index = np.unravel_index(best_index, table_shape)
    
    token_start, stem_start = best_index
    length = prefix_match_lengths[*best_index]
    
    return token_start, stem_start, length


def token2stemmingrule(token, stem):
    token, stem = token.strip(), stem.strip()
    
    # Align stem with token to extract prefix and suffix
    token_start, stem_start, length = align_stem(token, stem)
    
    token_prefix = token[:token_start]
    token_suffix = token[token_start + length:]
    
    stem_prefix = stem[:stem_start]
    stem_suffix = stem[stem_start + length:]
    
    return ((token_prefix, token_suffix), (stem_prefix, stem_suffix))


def apply_rule(token, rule):
    (token_prefix, token_suffix), (stem_prefix, stem_suffix) = rule
    if not (token.startswith(token_prefix) or token.endswith(token_suffix)):
        raise ValueError(f"({token_prefix}, {token_suffix}) -> ({stem_prefix}, {stem_suffix}) is invalid for {token}")
    
    if len(token_prefix) > 0:
        token = token[len(token_prefix):]
    
    if len(token_suffix) > 0:
        token = token[:-len(token_suffix)]
    
    return stem_prefix + token + stem_suffix
