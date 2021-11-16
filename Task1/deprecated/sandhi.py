"""Formulate sandhi rules from sandhied and unsandhied sequences.

   Data for task 1:
   - source: joint_sentence := sandhied sentence
   - target: t1_ground_truth := a list of tokens in unsandhied form

   For aligning source and target, we have either:
   - output of `tokenize()` := sandhied sentence split at split indices obtained from alignment
   - output of `to_syllables()` := source and target as lists of syllables
"""
# ---- main ----
import json
from helpers import load_data, load_sankrit_dictionary
from vocabulary import make_vocabulary
from pathlib import Path

# --------------

from logger import logger
import pprint

pp = pprint.PrettyPrinter(indent=4)

from subword import to_syllables

from collections import Counter


def find_syllables(joint_sent, ground_truth):
    """Prepare two lists of syllables."""
    return to_syllables(joint_sent), to_syllables(" ".join(ground_truth))


def find_sandhis(source, target):
    """Compare two lists of syllables to find Sandhi rules:"""
    assert len(target) >= len(source)
    logger.info("Source:")
    logger.debug(source)
    logger.info("Target:")
    logger.debug(target)

    # First we find the possible places where sandhis rules
    # could have been applied. These are simply locations
    # around whitespaces in the target sequence.

    # Therefore, we collect the previous and next syllable
    # of every whitespace, save them as a three-element
    # list, which will later be used for sequence
    # reconstruction.
    whitespaces = []
    indices = []
    for i, syll in enumerate(target):
        if syll == " ":
            # Hopefully nothing starts/ends with whitespaces!
            # For readability we write:
            prev_syll = target[i - 1]
            next_syll = target[i + 1]
            whitespaces.append(([prev_syll, " ", next_syll], (i - 1, i + 1)))
            indices.append(i - 1)
            indices.append(i + 1)
    logger.info("Collected syllables around whitespaces:")
    pp.pprint(whitespaces)

    repeated = [k for k, v in Counter(indices).items() if v >= 2]
    logger.info(f"Items affected by 2 rules:{repeated}")
    # Not every whitespace is a result of sandhi. We want to
    # know which of these whitespace-rules are used as sandhi.

    # The idea is to compare source and target lists one by one,
    # if match, save to reconstructed_sequence;
    # if don't match, replace syll in source with a whitespace rule
    # starting with this syll, etc.
    # Note down for which syllables we applied what rules
    # After the lists are processed, we are left with:
    # - used_rules
    # - reconstructed sequence
    # To make sure we didn't miss anything, we compare
    # our reconstructed sequence with the target sequence
    # Formulate sandhi rules, construct Sandhi objects
    # Return Sandhi objects and the source sentence with character
    # linked to Sandhi objects.

    used_rules, reconstructed_sequence = [], []

    i = 0  # source index
    j = 0  # target index

    while i < len(source):
        print(i, j)
        print(source[i], target[j])
        if target[j] != source[i]:  # Form disagrees
            if target[j][0] == source[i][0]:  # First character is the same
                rule = whitespaces.pop(0)
                if target[j] == rule[0][0]:  # Found matching rule
                    # Reconstruct
                    if j in repeated:
                        reconstructed_sequence.pop()
                    reconstructed_sequence.extend(rule[0])

                    # Check if ge --> ga I (one-three)
                    if (
                        source[i + 1] != " "
                        and source[i + 1][0] != rule[0][-1][0]
                        and source[i + 1][-1] != rule[0][-1][-1]
                    ):
                        if source[i + 1] == target[rule[1][1] + 1]:
                            print("case 4 (special case 1)")
                            used_rules.append(((i, [source[i]]), rule[0]))  # expansion
                            print(used_rules[-1])
                            # do not increment i
                            j = rule[1][1]
                        else:
                            used_rules.append(((i, [source[i]]), rule[0]))  # expansion
                            # do not increment i
                            j = rule[1][1]
                            print("case 1")
                    # Check if tanmu --> tat mu (two-three)
                    elif (
                        source[i + 1][0] == rule[0][-1][0]
                        or source[i + 1][-1] == rule[0][-1][-1]
                    ):
                        if source[i + 1] == target[rule[1][1] + 1]:
                            print("case 6 (special case 1)")
                            used_rules.append(((i, [source[i]]), rule[0]))  # expansion
                            print(used_rules[-1])
                            # do not increment i
                            j = rule[1][1]
                        else:
                            used_rules.append(
                                ((i, [source[i], source[i + 1]]), rule[0])
                            )  # expansion
                            i += 1
                            j = rule[1][1]
                            print(used_rules[-1])
                            print("case 2")
                    # Check if (three-three)
                    elif source[i + 1] == " ":
                        used_rules.append(
                            ((i, [source[i], source[i + 1], source[i + 2]]), rule[0])
                        )  # expansion
                        i += 2
                        j = rule[1][1]
                        print("case 3")

                    if (
                        source[i + 1][0] == target[j][0]
                        or source[i + 1][:2] == "r" + target[j][0]
                    ):
                        used_rules.append(((i, [source[i], source[i + 1]]), rule[0]))
                        i += 1
                    if j in repeated:
                        # reconstructed_sequence.pop()
                        i -= 1
                        j -= 1

            elif i + 1 == len(source) and target[j + 1]:
                rule = whitespaces.pop(0)
                # print(rule[0])
                if source[i - 1] == rule[0][0]:
                    reconstructed_sequence.pop()
                    reconstructed_sequence.extend(rule[0])
                    used_rules.append(((i - 1, [source[i - 1], source[i]]), rule[0]))

            elif (
                i + 1 <= len(source) and source[i + 1] == " "
            ):  # Is followed by a whitespace in source (meaning not sandhi?)
                # Check whitespace-rule
                rule = whitespaces.pop(0)
                logger.info(
                    f"source-{source[i]} and target-{target[j]}: {rule} apply to"
                    " previous syllable although forms match."
                )
                if (
                    rule[0][0] == source[i - 1]
                ):  # Rule matches previous token from source
                    # Keep track of rules used --- Tuple[Tuple, 3-element List]
                    used_rules.append(
                        ((i - 1, [source[i - 1], source[i], source[i + 1]]), rule[0])
                    )
                    print(f"used rules: {used_rules[-1]}")
                    j = rule[1][1] + 1  # end index in target sequence plus one
                    # Replace previous token, add syllables to reconstruction
                    reconstructed_sequence.pop()
                    reconstructed_sequence.extend(rule[0])
                    if j - 1 in repeated:
                        reconstructed_sequence.pop()

                else:
                    logger.warning("Something's wrong...")

                # Compare current 3 syllables with next rule
                assert i + 2 <= len(source)
                rule = whitespaces.pop(0)
                print(f"current rule: {rule[0]}")
                print(reconstructed_sequence)
                print([source[i], source[i + 1], source[i + 2]])
                if [source[i], source[i + 1]] == rule[0][:-1] and source[i + 2][
                    0
                ] == rule[0][-1][0]:
                    # This whitespace-rule is not sandhi
                    # Add syllables to reconstruction directly
                    reconstructed_sequence.extend(rule[0])
                    # Increment indices
                    if source[i + 1] != " ":  # plain split
                        used_rules.append(
                            ((i - 1, [source[i - 1], source[i]]), rule[0])
                        )
                        i += 1
                    else:
                        i += 2  # one less for final while-loop increment
                    j = rule[1][1]  # also save 1
                    if j in repeated:
                        reconstructed_sequence.pop()
                        i -= 1
                        j -= 1
            elif (
                j + 1 <= len(target) and target[j] == " " and target[j + 1] == source[i]
            ):
                rule = whitespaces.pop(0)
                # print([source[i - 1], " ", source[i]])
                # print(rule[0])
                # print("---------------------------------")
                if [source[i - 1], " ", source[i]] == rule[0]:
                    reconstructed_sequence.append(" ")
                    reconstructed_sequence.append(source[i])

                    used_rules.append(
                        ((i - 1, [source[i - 1], source[i], source[i + 1]]), rule[0])
                    )
                    j += 1
        elif source[i] != " " and target[j] != " ":
            reconstructed_sequence.append(source[i])

        else:  # If both are spaces, we have skipped one rule check because previous tokens agrees in form
            rule = whitespaces.pop(0)
            print(f"Skipped {rule}")
            print(f"source: {[source[i - 1], source[i], source[i + 1]]}")
            print(f"target: {rule[0]}")
            assert i + 1 <= len(source)
            if target[j + 1][0] != source[i + 1][0]:  # not properly aligned
                if target[j - 1][0] == source[i + 1][0]:
                    # Reconstruct
                    reconstructed_sequence.pop()  # remove duplicates
                    reconstructed_sequence.extend(rule[0])
                    # Save as used rule
                    used_rules.append(
                        (
                            (i + 1, [source[i + 1], source[i + 2], source[i + 3]]),
                            rule[0],
                        )
                    )
                    print(used_rules[-1])
                    # Increment indices
                    i += 1
                    j = rule[1][1]
            elif [source[i - 1], source[i], source[i + 1]] == rule[0]:
                # This whitespace-rule is not sandhi
                # Add syllables to reconstruction directly
                print(i, j)
                print("case x")
                print(reconstructed_sequence)

                reconstructed_sequence.pop()
                reconstructed_sequence.extend(rule[0])
                # Increment indices
                i += 1  # length is 3 but started with i-1, also save one for final while-loop increment
                j = rule[1][1]  # also save 1
            else:  # Sandhi
                print(i, j)
                print("case 5")
                reconstructed_sequence.pop()
                reconstructed_sequence.extend(rule[0])
                used_rules.append(
                    ((i - 1, [source[i - 1], source[i], source[i + 1]]), rule[0])
                )
                print(used_rules[-1])
                if not j + 1 in repeated:
                    i += 1
                    j = rule[1][1]

        i += 1
        j += 1

    logger.info("Used rules:")
    pp.pprint(used_rules)
    logger.info("Reconstructed sequence:")
    print(reconstructed_sequence)
    logger.info("Target sequence:")
    print(target)

    # Check if reconstructed sequence is different from target sequence
    # If so, identify other rules
    assert reconstructed_sequence == target

    # return sandhis
    return used_rules, reconstructed_sequence


class Sandhi(object):
    # __slots__ = ["current_char", "to_char", "next_char", "action"]

    def __init__(self, current_char, to_char, next_char, action):
        self.current_char = current_char
        self.to_char = to_char
        self.next_char = next_char
        self.action = action

    def __str__(self):
        attrs = vars(self)
        return " ".join("%s: %s" % item for item in attrs.items())


if __name__ == "__main__":
    with open("config.cfg") as cfg:
        config = json.load(cfg)

    translit = config["translit"]

    # Load data, data is a dictionary indexed by sent_id
    logger.info("Load data")
    train_data = load_data(config["train_path"], translit)
    eval_data = load_data(config["eval_path"], translit)

    # Display an example sentence

    logger.info(train_data[:3])

    # Make vocabulary
    # whitespaces are translated to '_' and treated as a normal character
    logger.info("Make vocab")
    vocabulary, char2index, index2char = make_vocabulary(train_data)

    logger.debug(f"{len(vocabulary)} chars in vocab:\n{vocabulary}")

    # find_sandhis(*train_data[0])

    for i in train_data[:50]:
        logger.info("\n-------------------------------------")
        logger.info(i[0])
        logger.info(" ".join(i[1]))
        sandhied_syllables, unsandhied_syllables = find_syllables(*i)
        rules, seq = find_sandhis(sandhied_syllables, unsandhied_syllables)