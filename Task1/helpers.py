import json
import pickle
import unicodedata

from uni2intern import unicode_to_internal_transliteration as to_intern

# Provided by task organizers alongside data

def unicode_normalize(dict_item):
    """
    It is recommended to use unicode normalization for the sanskrit strings to avoid potential normalization
    mismatches.
    Input:  Dict -> JSON object for a DCS sentence, an element from JSON List
    Output: Dict -> Same content as the input, but all the unicode strings normalized with NFC
    """
    dict_item["joint_sentence"] = unicodedata.normalize('NFC',dict_item["joint_sentence"])
    #joint sentence - input for tasks 1 and 3
    dict_item["segmented_sentence"] = unicodedata.normalize('NFC',dict_item["segmented_sentence"])
    #segmented sentence - input for task 2
    dict_item["t1_ground_truth"] = [[unicodedata.normalize('NFC',x) for x in inner]
                                    for inner in dict_item["t1_ground_truth"]]
    #ground truth for task 1 - word segmentation
    dict_item["t2_ground_truth"] = [[unicodedata.normalize('NFC',inner[0]), inner[1]]
                                    for inner in dict_item["t2_ground_truth"]]
    #ground truth for task 2 - morphological parsing
    dict_item["t3_ground_truth"] = [[[unicodedata.normalize('NFC',x[0]),
                                      unicodedata.normalize('NFC',x[1]), x[2]]
                                     for x in inner] for inner in dict_item["t3_ground_truth"] ]
    #ground truth for task 3 - word segmentation and morphological parsing

    return dict_item


# Provided by task organizers alongside data
# Modified to accomodate translit

def get_task1_IO(json_entry, translit=False):
    """
    Returns the input (as a string) and the groundtruth (as a list of segmented words) for the task 1:
    Word segmentation.
    Input: json object -> Json object for a DCS sentence
    Output: tuple(String, List) -> A tuple of the sentence (words joined with sandhi)
            and a flattened list of segmented words
    """

    if translit:
        flattened_list = [val for sublist in json_entry['t1_ground_truth'] for val in sublist]
        return (to_intern(json_entry["joint_sentence"]),flattened_list)
    else:
        flattened_list = [val for sublist in json_entry['t1_ground_truth'] for val in sublist]
        return (json_entry["joint_sentence"],flattened_list)


# Loading dataset from json file

def load_data(file_path, translit=False):
    with open(file_path, encoding='utf-8') as data_file:
        data = [unicode_normalize(item) for item in json.load(data_file)]
        # data = [get_task1_IO(sentence, translit) for sentence in data]
        data = {sentence['sent_id']:get_task1_IO(sentence, translit) for sentence in data}
        return data  # dictionary indexed by sent_id

# Load sanskrit dictionary

def load_sankrit_dictionary(file_path):
    with open(file_path, 'rb') as sanskrit_dict:
        return pickle.load(sanskrit_dict)

# Check input-output pair for a specific sentence

def check_specific_sent(data, sent_id):
    return data[sent_id]


