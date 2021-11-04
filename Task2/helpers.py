import json
import unicodedata

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

def get_task2_IO(json_entry):
    """
    Returns the input (as a string) and the groundtruth (as a list of tuples) for the task 2: Morphological parsing.
    Input: json object -> Json object for a DCS sentence
    Output: tuple(String, List of tuples) -> A tuple of the segmented sentence and a list of tuples with values (stem, morphological tag)
    """    
    
    list_of_tuples = [(sublist[0],sublist[1]) for sublist in json_entry['t2_ground_truth']] # Each inner list is converted to a tuple
                                                        # Each inner list represents the analysis for a word
                                                        # The tuple contains (stem, morphological-tag)
    
    
    return (json_entry["segmented_sentence"],list_of_tuples)


# Loading dataset from json file

def load_data(file_path):
    with open(file_path, encoding='utf-8') as data_file:
        data = [unicode_normalize(item) for item in json.load(data_file)]
        data = [get_task2_IO(sentence) for sentence in data]
        return data
        
