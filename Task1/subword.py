""" A simple syllable segmenter, only applicable
    if translit is true.
"""

from reconstruct_translit import vocals, consonants

V = set(vocals)
C = set(consonants)


def to_syllables(joint_sent):
    syllables = []
    syll = ""
    joint_sent = joint_sent + " "
    midx = len(joint_sent)
    for i, char in enumerate(joint_sent):

        if char == " ":
            if syll:
                syllables.append(syll)
            syllables.append(char)
            syll = ""
        elif char in V:
            if syllables and syllables[-1] == " " and not syll:  # Anlaut
                syllables.append(char)
            elif i + 1 < midx and joint_sent[i + 1] == " ":  # Auslaut
                syll += char
                syllables.append(syll)
                syll = ""
                # merge with prev syllable if syllable is just one C
                if len(syllables) > 1 and syllables[-2] in C:
                    syllables[-1] = "".join(syllables[-2:])
                    syllables.pop(-2)
            elif i + 2 < midx and joint_sent[i + 1] in C and joint_sent[i + 2] in C:
                syll += char
                syllables.append(syll)
                syll = ""
                # merge with prev syllable if syllable is just one C
                if len(syllables) > 1 and syllables[-2] in C:
                    syllables[-1] = "".join(syllables[-2:])
                    syllables.pop(-2)
            elif i + 2 < midx and joint_sent[i + 1] in C and joint_sent[i + 2] in V:
                syll += char
                syllables.append(syll)
                syll = ""
                # merge with prev syllable if syllable is just one C
                if len(syllables) > 1 and syllables[-2] in C:
                    syllables[-1] = "".join(syllables[-2:])
                    syllables.pop(-2)
            elif i + 1 < midx and joint_sent[i + 1] in C:
                syll += char

        elif char in C:
            if (
                not syll and char == "v" and i + 1 < midx and joint_sent[i + 1] == "y"
            ):  # vy
                syll += char
                continue
            if (
                not syll and char == "s" and i + 1 < midx and joint_sent[i + 1] == "y"
            ):  # sy

                syll += char
                continue
            if (
                not syll and char == "s" and i + 1 < midx and joint_sent[i + 1] == "T"
            ):  # sT

                syll += char
                continue
            elif char == "t" and i + 1 < midx and joint_sent[i + 1] == "t":  # tat tva
                syll += char
                syllables.append(syll)
                syll = ""
                continue
            elif i + 1 < midx and joint_sent[i + 1] in V:
                syll += char
            elif i + 2 < midx and joint_sent[i + 1] in C and joint_sent[i + 2] in C:
                if len(syll) >= 3:
                    syllables.append(syll)
                    syll = ""
                    syll += char
                else:
                    syll += char
            elif i + 2 < midx and joint_sent[i + 1] in C and joint_sent[i + 2] in V:
                syll += char
                # syllables.append(syll)
                # syll = ""
                # if len(syllables) > 1 and syllables[-2] in C:
                #     syllables[-1] = "".join(syllables[-2:])
                #     syllables.pop(-2)
            else:
                if syll:
                    syll += char

    return syllables[:-1]


if __name__ == "__main__":

    print(V)
    print(C)

    joint_sent = "kuraFgam_vAnaram_siMham_citram_vyAGram_tu_GAtayan"
    unsandhied = " ".join(
        ["kuraFgam", "vAnaram", "siMham", "citram", "vyAGram", "tu", "GAtayan"]
    )
    joint_sent = "GawasTayogam yogeSa tattva jQAnasya kAraNam"
    unsandhied = "Gawa sTa yogam yoga ISa tattva jQAnasya kAraNam"
    print(joint_sent)
    print(unsandhied)
    syllables = to_syllables(joint_sent)
    syllables_target = to_syllables(unsandhied)
    print(f"syllables: {syllables}")
    print(f"targets: {syllables_target}")
