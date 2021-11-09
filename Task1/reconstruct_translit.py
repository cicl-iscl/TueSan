import editdistance
import numpy as np

vocals = [
    "a",
    "'",  # a
    "e",
    "i",
    "o",
    "u",
    "A",  # ā
    "I",  # ī
    "U",  # ū
    "R",  # ṛ
    "L",  # ṝ
    "E",  # ai
    "O",  # au
    "?",  # ḷ or ḹ
]

consonants = [
    "K",  # kh
    "G",  # gh
    "F",  # ṅ
    "C",  # ch
    "J",  # jh
    "Q",  # ñ
    "W",  # ṭh
    "w",  # ṭ
    "X",  # ḍh
    "x",  # ḍ
    "N",  # ṇ
    "T",  # th
    "D",  # dh
    "P",  # ph
    "B",  # bh
    "S",  # ś
    "z",  # ṣ
    "M",  # ṃ
    "H",  # ḥ
    "b",
    "c",
    "d",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "v",
    "y",
]


def similarity(source, target):
    source_consonants = [char for char in source if char in consonants][:1]
    target_consonants = [char for char in target if char in consonants][:1]

    if len(source) <= 3 and source_consonants != target_consonants:
        return np.inf
    else:
        return editdistance.eval(source, target)


def reconstruct_unsandhied(tokens, allowed_words):
    reconstructed = []
    for token in tokens:
        similarities = np.array([similarity(token, word) for word in allowed_words])
        minimum_similarity = np.min(similarities)
        if minimum_similarity >= 5:
            continue

        best_allowed_indices = [
            index for index, sim in enumerate(similarities) if sim == minimum_similarity
        ]

        if len(best_allowed_indices) > 1:
            best_allowed_idx = min(
                best_allowed_indices,
                key=lambda idx: abs(len(allowed_words[idx]) - len(token)),
            )
        else:
            best_allowed_idx = best_allowed_indices[0]

        best_allowed_word = allowed_words[best_allowed_idx]
        if token.startswith("cA") and not best_allowed_word.startswith("c"):
            reconstructed.extend(["ca", best_allowed_word])
        else:
            reconstructed.append(best_allowed_word)

    return reconstructed
