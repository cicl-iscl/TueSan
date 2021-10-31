#!/usr/bin/env python3
import json
import time
from string import punctuation
from tqdm import tqdm
from edit_distance import SequenceMatcher as align
import numpy as np
import networkx as nx

import pickle
from pathlib import Path
from logger import logger
from utils import get_task_data, get_task1_IO

import pprint
pp = pprint.PrettyPrinter(indent=4)

# DATA_DIR = Path('/Users/jingwen/Desktop/challenges/sanskrit/Tüsan/')
DATA_DIR = Path('sanskrit/')  # relative path to local copy
# DATA_DIR = Path('/data/jingwen/sanskrit/') # on server

TRAIN_JSON = Path(DATA_DIR, 'wsmp_train.json')
TRAIN_GRAPHML = Path(DATA_DIR, 'final_graphml_train')
TRAIN_PICKLE = Path('train_dataset.pickle')

def read_json(jsonfile):
	with open(jsonfile, encoding="utf-8") as data_file:
		data = json.load(data_file)
		normalised_data = get_task_data(data)  # list of dictionary items from the training set
		return normalised_data

def get_task1_data(datapoint):
	sandhied, unsandhied_tokenized = get_task1_IO(datapoint)
	unsandhied = " ".join(unsandhied_tokenized)
	# Check punctuation: all punctuations are "'" (avagraha, stands for unpronounced 'a')
	# p = [char for char in sandhied if char in punctuation and char != "'"]
	# return p
	# Remove punctuation from input: for now don't, add "'" as vowel
	return sandhied, unsandhied, unsandhied_tokenized

def tokenize(sandhied, unsandhied):
	# Align unsandhied tokens with sandhied sentence
	# to get split points
	alignment = align(a = sandhied, b = unsandhied)
	alignment = alignment.get_matching_blocks()
	sandhied_indices, unsandhied_indices, _ = zip(*alignment)

	# Remove dummy indices
	sandhied_indices = sandhied_indices[:-1]
	unsandhied_indices = unsandhied_indices[:-1]

	# Find indices with spaces -> split
	split_indices = [index for index, char in enumerate(sandhied) if not char.strip()]

	# Find indices where sandhied/unsandhied sentence is
	# not aligned -> split here
	sandhied_pointer = 0
	unsandhied_pointer = 0

	for sandhied_index, unsandhied_index in zip(sandhied_indices, unsandhied_indices):
		if sandhied_pointer != sandhied_index or unsandhied_pointer != unsandhied_index:
			if sandhied_index + 1 not in split_indices and sandhied_index - 1 not in split_indices:
				split_indices.append(sandhied_index)
				sandhied_pointer = sandhied_index
				unsandhied_pointer = unsandhied_index

		sandhied_pointer += 1
		unsandhied_pointer += 1

	split_indices = [0] + list(sorted(set(split_indices)))

	# Tokenize sandhied sentence:
	# Split at split indices
	split_indices = zip(split_indices, split_indices[1:] + [None])
	tokens = [sandhied[start:stop] for start, stop in split_indices]
	tokens = [token.strip() for token in tokens if not token.isspace()]

	return tokens

def remove_trailing_syllables(tokens, unsandhied_tokenized):
	if len(tokens) > len(unsandhied_tokenized) and \
		len(tokens) > 2 and \
		unsandhied_tokenized[-1].endswith(tokens[-1]) and \
		unsandhied_tokenized[:3] == tokens[-2].strip()[:3]:
		tokens = tokens[:-1]

	return tokens

def merge_single_letters(tokens):
	merged_tokens = []
	skip_next = False

	for index, token in enumerate(tokens):
		if skip_next:
			skip_next = False
			continue

		token = token.strip()
		if len(token.strip()) == 1 and index < len(tokens) - 1:
			merged_token = token + tokens[index + 1].strip()
			merged_tokens.append(merged_token)
			skip_next = True
		else:
			merged_tokens.append(token)

	return merged_tokens

def make_labels(tokens):
	combined_string = "".join(tokens)

	offsets = np.array([len(token) for token in tokens])
	split_indices = np.cumsum(offsets) - 1

	labels = np.zeros(len(combined_string), dtype=np.int8)
	labels[split_indices] = 1

	return combined_string, labels

def get_allowed_words(sent_id):
	graphml_path = Path(TRAIN_GRAPHML, str(str(sent_id) + ".graphml"))
	with open(graphml_path, mode='rb') as graph_input:
		graph = nx.read_graphml(graph_input)
		allowed_words = [node[1]['word'] for node in graph.nodes(data = True)]

	return list(sorted(set(allowed_words)))


def load_data(picklefile):
	dataset = []
	if TRAIN_PICKLE.is_file():
		with open(picklefile, 'rb') as data:
			dataset = pickle.load(data)
	else:
		normalised_training_data = read_json(TRAIN_JSON)
		for sent in tqdm(normalised_training_data):
			# p = get_task1_datent)
			# if p: logger.debug(p)
			sent_id = sent['sent_id']
			sandhied, unsandhied, unsandhied_tokenized = get_task1_data(sent)

			# Filter sents where sandhied >> unsandhied
			if len(sandhied) / len(unsandhied) > 1.5:
				# logger.warning(f'sandhied: {sandhied}\n')
				# logger.warning(f'unsandhied: {unsandhied}\n')
				continue

			# ---- tokenize ----
			tokens = tokenize(sandhied, unsandhied)
			tokens = remove_trailing_syllables(tokens, unsandhied_tokenized)
			tokens = merge_single_letters(tokens)
			sandhied_merged, labels = make_labels(tokens)
			allowed_words = get_allowed_words(sent_id)

			# ---- construct datapoint ----
			datapoint = {
				'sandhied_merged': sandhied_merged,
				'labels': labels,
				'tokens': tokens,
				'allowed_words': allowed_words,
				'unsandhied': unsandhied.split(),
				'sent_id': sent_id
			}

			dataset.append(datapoint)

		with open(picklefile, 'wb') as out:
			pickle.dump(dataset, out)
	return dataset



if __name__ == '__main__':

	logger.info("Reading and normalising training data ...")
	s1 = time.time()
	normalised_training_data = read_json(TRAIN_JSON)
	logger.debug(normalised_training_data[0])
	logger.info(f"Took {time.time()-s1:.2f} seconds.")

	s2 = time.time()
	dataset = load_data(TRAIN_PICKLE)
	logger.info(f"Took {time.time()-s2:.2f} seconds to prepare task 1 data.")

	logger.debug(pp.pprint(dataset[0]))
