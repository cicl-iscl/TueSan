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
from edit_distance import SequenceMatcher as align

from tqdm import tqdm

def find_syllables(joint_sent, ground_truth):
    """Prepare two lists of syllables."""
    return to_syllables(joint_sent), to_syllables(" ".join(ground_truth))


def find_sandhis(source, target):
    """Compare two lists of syllables to find Sandhi rules:"""
    #assert len(target) >= len(source)
    if len(source) >= len(target):
        return []
    #logger.info("Source:")
    #logger.debug(source)
    #logger.info("Target:")
    #logger.debug(target)
    
    used_rules = []
    
    alignment = align(a=source, b=target)
    alignment = alignment.get_matching_blocks()
    source_indices, target_indices, _ = zip(*alignment)
    
    #logger.info("Source Indices:")
    #logger.debug(source_indices)
    #logger.info("Target Indices:")
    #logger.debug(target_indices)
    
    if source_indices[0] != 0:
        used_rules.append((source[:source_indices[0]], target[:target_indices[0]]))
    
    for k in range(len(source_indices)):
        s_start_idx = source_indices[k]
        s_stop_idx = source_indices[k+1] if k + 1 < len(source_indices) else None
        t_start_idx = target_indices[k]
        t_stop_idx = target_indices[k+1] if k + 1 < len(source_indices) else None
        
        if s_stop_idx is None or s_stop_idx - s_start_idx == 1:
            used_rules.append((source[s_start_idx:s_stop_idx], target[t_start_idx:t_stop_idx]))
        else:
            used_rules.append((source[s_start_idx], target[t_start_idx]))
            used_rules.append((source[s_start_idx+1:s_stop_idx], target[t_start_idx+1:t_stop_idx]))
    
    #logger.info("Used rules")
    #logger.debug(used_rules)
    
    reconstructed_sequence = []
    for rule in used_rules:
        reconstructed_sequence += rule[1]
    reconstructed_sequence = "".join(reconstructed_sequence)
    
    if reconstructed_sequence != target:
        logger.debug(source)
        logger.debug(used_rules)
        logger.debug(reconstructed_sequence)
        logger.debug(target)
        logger.info(" ")
        
        return []
    
    return used_rules
    
    #logger.info(" ")
            
        
        

    # Check if reconstructed sequence is different from target sequence
    # If so, identify other rules
    #assert reconstructed_sequence == target

    # return sandhis
    #return used_rules, reconstructed_sequence


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
    all_rules = set()

    for i in tqdm(train_data[:]):
        #logger.info("\n-------------------------------------")
        #logger.info(i[0])
        #logger.info(" ".join(i[1]))
        #sandhied_syllables, unsandhied_syllables = find_syllables(*i)
        source = i[0]
        target = " ".join(i[1])
        used_rules = find_sandhis(source, target)
        
        all_rules.update(set(used_rules))
        
    all_rules = [rule for rule in all_rules if rule[0] != rule[1]]
    
    logger.info(all_rules)
    logger.info(len(all_rules))
        
