"""Create a lookup dictionary for words in the training
   and development datasets from graphml files.
"""
import os
import re
import pickle
import json
import unicodedata
import networkx as nx
from pathlib import Path
from logger import logger
from tqdm import tqdm

import pprint
pp = pprint.PrettyPrinter(indent=4)


# DATA_DIR = Path('sanskrit/')  # relative path to local copy
DATA_DIR = Path('/data/jingwen/sanskrit/') # on server

TRAIN_JSON = Path(DATA_DIR, 'wsmp_train.json')
TRAIN_GRAPHML = Path(DATA_DIR, 'final_graphml_train')
DEV_JSON = Path(DATA_DIR, 'wsmp_dev.json')
DEV_GRAPHML = Path(DATA_DIR, 'graphml_dev')


def get_all_nodes(sent_id, train=True):
	"""
	Returns the list of all nodes from the corresponding graphml file
	Input: integer -> sentence ID
	Output: list of tuples -> A list of tuples representing nodes where each tuple contains two values. node_id (integer) and node attributes (dictionary)
	"""
	if train:
		graphml_path = Path(TRAIN_GRAPHML, f"{str(sent_id)}.graphml") # graphml files for training are in final_graphml_train/
	else:
		graphml_path = Path(DEV_GRAPHML, f"{str(sent_id)}.graphml")																				# graphml files for dev are in graphml_dev/

	graph_input = open(graphml_path, mode='rb')

	graph = nx.read_graphml(graph_input) # reading the graph structure from the file contents

	nodes = list(graph.nodes(data = True)) # generating the list of nodes of the graph

	return nodes

def get_ground_truth_nodes(json_entry):
	"""
	Returns a tuple which has two lists: list of the ground truth nodes from the corresponding graphml file in nested and flattened format
	Input: JSON object -> json object for a DCS sentence
	Output: tuple of two lists -> A nested list of tuples representing nodes where each tuple contains two values. node_id (integer) and node attributes (dictionary)
		                      A flattened list of tuples representing nodes where each tuple contains two values. node_id (integer) and node attributes (dictionary)
	"""

	all_nodes = get_all_nodes(json_entry["sent_id"]) # get all the possible nodes from the graph

	graph_nodes = [[[node for node in all_nodes if str(node[0]) == str(val)][0] # nodes list by comparing with graphml node ids
					for val in sublist]                                         # each inner list represents the nodes for each of the word
				   for sublist in json_entry["graphml_node_ids"]]               # outer list representing chunk

	flattened_graph_nodes = [[node for node in all_nodes if str(node[0]) == str(val)][0] # flattened nodes by comparing with graphml node ids
							 for sublist in json_entry["graphml_node_ids"]               # inner list
							 for val in sublist]                                         # outer list representing chunk

	return (graph_nodes, flattened_graph_nodes)


def unicode_normalize(dict_item):
	"""
	It is recommended to use unicode normalization for the sanskrit strings to avoid potential normalization mismatches.
	Input:  Dict -> JSON object for a DCS sentence, an element from JSON List
	Output: Dict -> Same content as the input, but all the unicode strings normalized with NFC
	"""
	dict_item["joint_sentence"] = unicodedata.normalize('NFC',dict_item["joint_sentence"]) #joint sentence - input for tasks 1 and 3
	dict_item["segmented_sentence"] = unicodedata.normalize('NFC',dict_item["segmented_sentence"])  #segmented sentence - input for task 2
	dict_item["t1_ground_truth"] = [[unicodedata.normalize('NFC',x) for x in inner] for inner in dict_item["t1_ground_truth"]]   #ground truth for task 1 - word segmentation
	dict_item["t2_ground_truth"] = [[unicodedata.normalize('NFC',inner[0]), inner[1]] for inner in dict_item["t2_ground_truth"]] #ground truth for task 2 - morphological parsing
	dict_item["t3_ground_truth"] = [[[unicodedata.normalize('NFC',x[0]), unicodedata.normalize('NFC',x[1]), x[2]] for x in inner] for inner in dict_item["t3_ground_truth"] ]   #ground truth for task 3 - word segmentation and morphological parsing

	return dict_item


def load_data(data_file):
	with open(data_file, encoding='utf-8') as df:
		data = json.load(df)

	normalised_data = list()
	for item in data:
		dict_item = unicode_normalize(item)
		normalised_data.append(dict_item)

	return normalised_data


# ---- helper method to check graphml structure ----
def check_specific_sent(normalised_data, sent_id):
	for entry in normalised_data:
		if entry['sent_id'] == sent_id:
			logger.info(entry["graphml_node_ids"])

			full_nodes = get_all_nodes(sent_id)
			nodes, flattened_nodes = get_ground_truth_nodes(entry)
			# pp.pprint(nodes)
			logger.debug('==============================================')
			pp.pprint(full_nodes)
			# logger.debug('==============================================')
			# pp.pprint(flattened_nodes)
			# logger.debug('==============================================')



def get_stems(dictionary, word):
	return [a[0] for a in dictionary[word]]

def get_morphs(dictionary, word):
	return [a[1] for a in dictionary[word]]

def get_analyses(dictionary, word):
	return list(dictionary[word])

if __name__ == '__main__':

	words = set()
	dictionary = {}

	# ---- check structure ----
	dataset = load_data(TRAIN_JSON)
	# check_specific_sent(dataset, sent_id=6)

	train_dev = [TRAIN_JSON, DEV_JSON]

	for json_file in train_dev:
		dataset = load_data(json_file)  # load normalised data
		for entry in tqdm(dataset):
			nodes = get_all_nodes(entry['sent_id'])
			for node_id, node in nodes:
				word = node['word']
				stem = node['stem']
				morph = node['morph']
				if word not in words:
					words.add(node['word'])
					dictionary[word]=set()
				dictionary[word].add(tuple((stem, morph)))

	with open('words.pickle', 'wb') as out:
		pickle.dump(words, out)
	with open('dictionary.pickle', 'wb') as out:
		pickle.dump(dictionary, out)

	# --- test saved dictionary lookup ----
	with open('dictionary.pickle', 'rb') as data:
		dictionary = pickle.load(data)

		pp.pprint(dictionary)

		# ss = get_stems(dictionary, 'asya')
		# ms = get_morphs(dictionary, 'asya')
		# analyses = get_analyses(dictionary, 'asya')
		# logger.info(ss)
		# logger.info(ms)
		# logger.info(analyses)

