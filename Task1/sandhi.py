"""Formulate sandhi rules from sandhied and unsandhied sequences.
"""
import json

from helpers import load_data, load_sankrit_dictionary
from vocabulary import make_vocabulary


from pathlib import Path
import time
from logger import logger
import pprint

pp = pprint.PrettyPrinter(indent=4)

from subword import to_syllables


def find_syllables(joint_sent, ground_truth):
    sandhied_syllables = to_syllables(joint_sent)
    logger.debug(sandhied_syllables)
    unsandhied = " ".join(ground_truth)
    unsandhied_syllables = to_syllables(unsandhied)
    logger.debug(unsandhied_syllables)
    return sandhied_syllables, unsandhied_syllables


def find_sandhis(sandhied_syllables, unsandhied_syllables):
    """Compare two lists of syllables to find Sandhi rules:"""
    assert len(unsandhied_syllables) >= len(sandhied_syllables)
    sandhis = []
    i = 0
    j = 0
    max_idx = len(sandhied_syllables)
    while i < max_idx:
        if unsandhied_syllables[i] == sandhied_syllables[i]:
            for char in sandhied_syllables[i]:
                sandhis.append(Sandhi(char, char, "", "keep"))
            i += 1
            j = i
        elif unsandhied_syllables[i][:-1] == sandhied_syllables[i][:-1]:
            for char in sandhied_syllables[:-1]:
                sandhis.append(Sandhi(char, char, "", "keep"))
            if i + 1 < max_idx:
                sandhis.append(
                    Sandhi(
                        sandhied_syllables[i][:-1],
                        unsandhied_syllables[i][:-1],
                        sandhied_syllables[i + 1][0],
                        "replace",
                    )
                )
            i += 1
            j = i
        i += 1

    return sandhis


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

    for i in train_data[:5]:
        logger.info("\n-------------------------------------")
        logger.info(i[0])
        logger.info(" ".join(i[1]))
        sandhied_syllables, unsandhied_syllables = find_syllables(*i)
        sandhis = find_sandhis(sandhied_syllables, unsandhied_syllables)
        for sandhi in sandhis:
            logger.info(sandhi)
