import random
import editdistance
import numpy as np

from pathlib import Path
from tqdm import tqdm
from prepare_data import load_data
from logger import logger

TRAIN_PICKLE = Path('train_dataset.pickle')
task1_dataset = load_data(TRAIN_PICKLE)

VOWELS = ['a','e', 'i', 'o', 'u', 'ā', 'ī', 'ū', 'ṛ', 'ṝ', "'"]
CONSONANTS = [
			'b', 'c', 'd', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r',
			's', 't', 'v', 'y', 'ñ', 'ś', 'ḍ', 'ṃ', 'ṅ', 'ṇ', 'ṣ', 'ṭ', 'ḥ'
			 ]

def similarity(source, target):
	source_consonants = [char for char in source if char in CONSONANTS][:1]
	target_consonants = [char for char in target if char in CONSONANTS][:1]

	if len(source) <= 3 and source_consonants != target_consonants:
		return np.inf
	else:
		return editdistance.eval(source, target)


def reconstruct_unsandhied(tokens, allowed_words):
	reconstructed = []
	for token in tokens:
		similarities = np.array([similarity(token, word) for word in allowed_words])
		minimum_similarity = np.min(similarities)
		if minimum_similarity >= 5:
			continue

		best_allowed_indices = [
			index for index, sim in enumerate(similarities)
			if sim == minimum_similarity
		]

		if len(best_allowed_indices) > 1:
			best_allowed_idx = min(best_allowed_indices,
			key=lambda idx: abs(len(allowed_words[idx]) - len(token)))
		else:
			best_allowed_idx = best_allowed_indices[0]

		best_allowed_word = allowed_words[best_allowed_idx]
		if token.startswith('cā') and not best_allowed_word.startswith('c'):
			reconstructed.extend(['ca', best_allowed_word])
		else:
			reconstructed.append(best_allowed_word)

	return reconstructed

def get_next(idx, zipped):
	next_pred, next_true = '', ''
	if idx + 1 >= len(zipped):
		logger.warning('end token has error.')
	else:
		next_pred, next_true = zipped[idx+1]
	return next_pred, next_true


if __name__ == '__main__':

	# ---- test ----
	s1 = 'garhitāv'
	s2 = 'garhitau'

	logger.info(f'{s1}:\t{[char for char in s1 if char in CONSONANTS]}')
	logger.info(f'{s2}:\t{[char for char in s2 if char in CONSONANTS]}')

	logger.info(f'similarity:\t{similarity(s1, s2)}')

	# ---- error analysis ----
	wrong_a_errors = 0
	wrong_i_errors = 0
	wrong_u_errors = 0
	wrong_e_errors = 0
	wrong_o_errors = 0
	wrong_ai_errors = 0
	wrong_au_errors = 0
	missing_h_errors = 0
	consonant_errors = 0
	propagated_errors = 0
	missing_word_errors = 0
	other_errors = 0

	dataset = task1_dataset

	# for dp in tqdm(dataset[:5000]):
	for dp in tqdm(dataset):
		reconstructed = reconstruct_unsandhied(dp['tokens'], dp['allowed_words'])
		unsandhied = dp['unsandhied']

		if reconstructed == unsandhied:
			continue

		if len(reconstructed) != len(unsandhied):
			missing_word_errors += 1
			logger.debug('---- MISSING WORD ----')
			logger.warning(f'reconstructed:\t{reconstructed}')
			logger.warning(f'unsandhied:\t{unsandhied}')
			continue

		zipped = list(zip(reconstructed, unsandhied))
		for i, (pred_token, true_token) in enumerate(zipped):

			if pred_token == true_token:
				continue

			next_pred_token, next_true_token = get_next(i, zipped)


			if len(pred_token) == len(true_token) and pred_token[-1] == 'ā':
				wrong_a_errors +=1
				logger.debug('---- wrong-ā-error ----')
				logger.warning(f'pred:\t{pred_token}')
				logger.warning(f'true:\t{true_token}')
				logger.info(f'with next pred:\t{pred_token}\t{next_pred_token}')
				logger.info(f'with next true:\t{true_token}\t{next_true_token}\n')
			elif len(pred_token) + 1 == len(true_token) and next_true_token.startswith(('a', 'ā')):
				wrong_a_errors +=1
				logger.debug('---- wrong-ā-error ----')
				logger.warning(f'pred:\t{pred_token}')
				logger.warning(f'true:\t{true_token}')
				logger.info(f'with next pred:\t{pred_token}\t{next_pred_token}')
				logger.info(f'with next true:\t{true_token}\t{next_true_token}\n')
			elif len(pred_token) == len(true_token) and pred_token[-1] == 'ī':
				wrong_i_errors +=1
				logger.debug('---- wrong-ī-error ----')
				logger.warning(f'pred:\t{pred_token}')
				logger.warning(f'true:\t{true_token}')
				logger.info(f'with next pred:\t{pred_token}\t{next_pred_token}')
				logger.info(f'with next true:\t{true_token}\t{next_true_token}\n')
			elif len(pred_token) + 1 == len(true_token) and next_true_token.startswith(('i', 'ī')):
				wrong_i_errors +=1
				logger.debug('---- wrong-ī-error ----')
				logger.warning(f'pred:\t{pred_token}')
				logger.warning(f'true:\t{true_token}')
				logger.info(f'with next pred:\t{pred_token}\t{next_pred_token}')
				logger.info(f'with next true:\t{true_token}\t{next_true_token}\n')
			elif len(pred_token) == len(true_token) and pred_token[-1] == 'ū':
				wrong_u_errors +=1
				logger.debug('---- wrong-ū-error ----')
				logger.warning(f'pred:\t{pred_token}')
				logger.warning(f'true:\t{true_token}')
				logger.info(f'with next pred:\t{pred_token}\t{next_pred_token}')
				logger.info(f'with next true:\t{true_token}\t{next_true_token}\n')
			elif len(pred_token) + 1 == len(true_token) and next_true_token.startswith(('u', 'ū')):
				wrong_u_errors +=1
				logger.debug('---- wrong-ū-error ----')
				logger.warning(f'pred:\t{pred_token}')
				logger.warning(f'true:\t{true_token}')
				logger.info(f'with next pred:\t{pred_token}\t{next_pred_token}')
				logger.info(f'with next true:\t{true_token}\t{next_true_token}\n')
			elif len(pred_token) == len(true_token) and pred_token[-1] == 'e':
				wrong_e_errors +=1
				logger.debug('---- wrong-e-error ----')
				logger.warning(f'pred:\t{pred_token}')
				logger.warning(f'true:\t{true_token}')
				logger.info(f'with next pred:\t{pred_token}\t{next_pred_token}')
				logger.info(f'with next true:\t{true_token}\t{next_true_token}\n')
			elif len(pred_token) != len(true_token) and pred_token[-1] == 'o':
				wrong_o_errors +=1
				logger.debug('---- wrong-o-error ----')
				logger.warning(f'pred:\t{pred_token}')
				logger.warning(f'true:\t{true_token}')
				logger.info(f'with next pred:\t{pred_token}\t{next_pred_token}')
				logger.info(f'with next true:\t{true_token}\t{next_true_token}\n')
			elif len(pred_token) != len(true_token) and pred_token[-2:] == 'ai':
				wrong_ai_errors +=1
				logger.debug('---- wrong-ai-error ----')
				logger.warning(f'pred:\t{pred_token}')
				logger.warning(f'true:\t{true_token}')
				logger.info(f'with next pred:\t{pred_token}\t{next_pred_token}')
				logger.info(f'with next true:\t{true_token}\t{next_true_token}\n')
			elif len(pred_token) == len(true_token) and pred_token[-2:] == 'au':
				wrong_au_errors +=1
				logger.debug('---- wrong-au-error ----')
				logger.warning(f'pred:\t{pred_token}')
				logger.warning(f'true:\t{true_token}')
				logger.info(f'with next pred:\t{pred_token}\t{next_pred_token}')
				logger.info(f'with next true:\t{true_token}\t{next_true_token}\n')
			elif len(pred_token) + 1 == len(true_token) and true_token[-1] == 'ḥ':
				missing_h_errors += 1
				logger.debug('---- missing-ḥ-error ----')
				logger.warning(f'pred:\t{pred_token}')
				logger.warning(f'true:\t{true_token}')
				logger.info(f'with next pred:\t{pred_token}\t{next_pred_token}')
				logger.info(f'with next true:\t{true_token}\t{next_true_token}\n')
			elif pred_token[0] != true_token[0]:
				propagated_errors +=1
				logger.debug('---- propagated errors ----')
				logger.warning(f'pred:\t{pred_token}')
				logger.warning(f'true:\t{true_token}')
				logger.info(f'with next pred:\t{pred_token}\t{next_pred_token}')
				logger.info(f'with next true:\t{true_token}\t{next_true_token}\n')
			elif len(pred_token) == len(true_token) and pred_token[-1] in CONSONANTS and true_token[-1] in CONSONANTS:
				consonant_errors +=1
				logger.debug('---- consonant errors ----')
				logger.warning(f'pred:\t{pred_token}')
				logger.warning(f'true:\t{true_token}')
				logger.info(f'with next pred:\t{pred_token}\t{next_pred_token}')
				logger.info(f'with next true:\t{true_token}\t{next_true_token}\n')
			else:
				other_errors += 1
				logger.debug('---- unspecified error ----')
				logger.warning(f'pred:\t{pred_token}')
				logger.warning(f'true:\t{true_token}')
				logger.info(f'with next pred:\t{pred_token}\t{next_pred_token}')
				logger.info(f'with next true:\t{true_token}\t{next_true_token}\n')
				logger.debug('=================================\n')
				logger.debug(f"tokens:\t{dp['tokens']}")
				logger.debug(f"reconstructed:\t{reconstructed}")
				logger.debug(f"unsandhied:\t{dp['unsandhied']}\n")
				logger.debug('==================================')


	# total_errors = wrong_a_errors + missing_h_errors + missing_word_errors + other_errors
	total_errors = sum([
			wrong_a_errors,
			wrong_i_errors,
			wrong_u_errors,
			wrong_e_errors,
			wrong_o_errors,
			wrong_ai_errors,
			wrong_au_errors,
			missing_h_errors,
			# propagated_errors,
			consonant_errors,
			missing_word_errors,
			other_errors,
		]) - propagated_errors

	# ---- (long) simple vowel ----
	logger.info(f"Wrong a errors: {wrong_a_errors}")
	logger.info(f"Wrong i errors: {wrong_i_errors}")
	logger.info(f"Wrong u errors: {wrong_u_errors}")

	# ---- dipthong ----
	logger.info(f"Wrong e errors: {wrong_e_errors}")
	logger.info(f"Wrong o errors: {wrong_o_errors}")
	logger.info(f"Wrong ai errors: {wrong_ai_errors}")
	logger.info(f"Wrong au errors: {wrong_au_errors}")

	# ---- Hauchlaut ----
	logger.info(f"Missing h errors: {missing_h_errors}")
	# Sometimes the vowel before this ḥ is elongated
	# Can also introduce changes to the initial part of the next word

	# ---- consonants ----
	logger.info(f"Consonant errors: {consonant_errors}")

	logger.info(f"Propagated errors: {propagated_errors}")
	logger.info(f"Missing word errors: {missing_word_errors}")
	logger.info(f"Other errors: {other_errors}")
	logger.info(f"Total errors: {total_errors}")
	#print(errors)
