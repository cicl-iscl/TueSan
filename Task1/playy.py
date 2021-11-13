from logger import logger
import pprint

pp = pprint.PrettyPrinter(indent=4)

source = [
    "ni",
    "drA",
    "nte",
    " ",
    "Bo",
    "ja",
    "nA",
    "nte",
    " ",
    "ca",
    " ",
    "di",
    "vA",
    "nte",
    " ",
    "ca",
    " ",
    "di",
    "ne",
    " ",
    "di",
    "ne",
]
target = [
    "ni",
    "drA",
    " ",
    "a",
    "nte",
    " ",
    "Bo",
    "ja",
    "nA",
    " ",
    "a",
    "nte",
    " ",
    "ca",
    " ",
    "di",
    "vA",
    " ",
    "a",
    "nte",
    " ",
    "ca",
    " ",
    "di",
    "ne",
    " ",
    "di",
    "ne",
]

if __name__ == "__main__":

    whitespaces = []  # places where changes occur
    for i, syll in enumerate(target):
        if syll == " ":
            prev_syll = target[i - 1]  # hopefully nothing starts/ends with whitespaces
            next_syll = ""
            if i + 1 <= len(target):  # safety: has next
                next_syll = target[i + 1]
                whitespaces.append(([prev_syll, " ", next_syll], (i - 1, i + 1)))
    pp.pprint(whitespaces)

    used_rules = []
    reconstructed_sequence = []
    i = 0  # source index
    j = 0  # target index
    while i < len(source):
        print(i, j)
        if target[j] != source[i]:  # form disagrees
            if (
                i + 1 <= len(source) and source[i + 1] == " "
            ):  # is followed by a whitespace in source (meaning not sandhi?)
                # check rule
                rule = whitespaces.pop(0)
                print(f"rule: {rule}")
                if (
                    rule[0][0] == source[i - 1]
                ):  # if rule matches previous token from source
                    # keep track of rules used
                    # Tuple[Tuple, 3-element List]
                    used_rules.append(((i - 1, source[i - 1]), rule[0]))
                    # replace previous token
                    reconstructed_sequence.pop()
                    reconstructed_sequence.extend(rule[0])
                    # increment indices
                    j = rule[1][1] + 1  # end index in target sequence plus one
                else:
                    logger.warning("Something's wrong...")

                # compare current 3 syllables with next rule
                assert i + 2 <= len(source)
                rule = whitespaces.pop(0)
                if [source[i], source[i + 1], source[i + 2]] == rule[0]:
                    # this is not a sandhi rule
                    # add syllables
                    reconstructed_sequence.extend(rule[0])
                    # increment indices
                    i += 2
                    j = rule[1][1]
        elif source[i] != " " and target[j] != " ":
            reconstructed_sequence.append(source[i])

        else:  # if both are spaces, we have skipped one rule check because prev agrees in form
            rule = whitespaces.pop(0)
            print(f"skipped {rule}")
            print(f"{[source[i - 1], source[i], source[i + 1]]}")
            print(f"{rule[0]}")
            assert i + 1 <= len(source)
            if [source[i - 1], source[i], source[i + 1]] == rule[0]:
                # this is not a sandhi rule
                # add syllables
                reconstructed_sequence.pop()
                reconstructed_sequence.extend(rule[0])
                # increment indices
                i += 1
                j = rule[1][1]
            else:  # this is a sandhi rule
                used_rules.append(((i - 1, source[i - 1]), rule[0]))
                i += 1
                j = rule[1][1]

        i += 1
        j += 1

    print(used_rules)
    print(reconstructed_sequence)

    # if reconstructed sequence is different from target sequence, find other sandhi rules
