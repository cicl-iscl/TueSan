import unicodedata

from logger import logger
import pprint
pp = pprint.PrettyPrinter(indent=4)


def get_task_data(data):
	"""
	Returns the unicode normalised data from the given data
	Input: Data list
	Output: List of dictionary items
	"""
	normalised_data = list()
	for (item) in data:  # iterate over the list of json data to operate on the values
		dict_item = unicode_normalize(item)  # recommended to use unicode normalization
		# for the sanskrit strings to avoid potential normalization mismatches
		normalised_data.append(dict_item)  # List containing all the entries in the json file, but after normalization

	return normalised_data


def get_task1_IO(json_entry):
	"""
	Returns the input (as a string) and the groundtruth (as a list of segmented words) for the task 1:
	Word segmentation.
	Input: json object -> Json object for a DCS sentence
	Output: tuple(String, List) -> A tuple of the sentence (words joined with sandhi)
			and a flattened list of segmented words
	"""
	flattened_list = [val for sublist in json_entry['t1_ground_truth'] for val in sublist]
	# Flatten the nested list to a single linear list

	return (json_entry["joint_sentence"],flattened_list)


def unicode_normalize(dict_item):
	"""
	It is recommended to use unicode normalization for the sanskrit strings to avoid potential normalization
	mismatches.
	Input:  Dict -> JSON object for a DCS sentence, an element from JSON List
	Output: Dict -> Same content as the input, but all the unicode strings normalized with NFC
	"""
	if "joint_sentence" in dict_item.keys():
		dict_item["joint_sentence"] = unicodedata.normalize('NFC',dict_item["joint_sentence"])
		# joint sentence - input for tasks 1 and 3
	if "segmented_sentence" in dict_item.keys():
		dict_item["segmented_sentence"] = unicodedata.normalize('NFC',dict_item["segmented_sentence"])
		# segmented sentence - input for task 2
	if "t1_ground_truth" in dict_item.keys():
		dict_item["t1_ground_truth"] = [[unicodedata.normalize('NFC',x) for x in inner]
									for inner in dict_item["t1_ground_truth"]]
		# ground truth for task 1 - word segmentation
	if "t2_ground_truth" in dict_item.keys():
		dict_item["t2_ground_truth"] = [[unicodedata.normalize('NFC',inner[0]), inner[1]]
									for inner in dict_item["t2_ground_truth"]]
	# ground truth for task 2 - morphological parsing
	if "t3_ground_truth" in dict_item.keys():
		dict_item["t3_ground_truth"] = [[[unicodedata.normalize('NFC',x[0]),
									  unicodedata.normalize('NFC',x[1]), x[2]]
									 for x in inner] for inner in dict_item["t3_ground_truth"] ]
	#ground truth for task 3 - word segmentation and morphological parsing

	return dict_item


def check_specific_sent(normalised_data, sent_id):
	"""For Task 1:
		 Given sent_id, display the groundtruth tokens and the nodes from the graph.
	"""
	for entry in normalised_data:
		if entry['sent_id'] == sent_id:
			t1_groundtruth = get_task1_IO(entry)
			logger.debug(entry['joint_sentence'])
			logger.debug(t1_groundtruth)

			# logger.info(entry["graphml_node_ids"])

			# full_nodes = get_all_nodes(sent_id)
			# nodes = get_ground_truth_nodes(entry)
			# logger.debug(pp.pprint(nodes))
			# pp.pprint(full_nodes)
			# logger.debug(pp.pprint(full_nodes))
