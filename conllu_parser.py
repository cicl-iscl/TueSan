"""Utilities for parsing Digital Corpus of Sanskrit (DCS) conllu files.
"""

import re
from pathlib import Path
from itertools import chain

import json

from logger import logger
import pprint
pp = pprint.PrettyPrinter(indent=4)

# DATA_DIR = Path('sanskrit')
DATA_DIR = Path('data','jingwen', 'sanskrit')  # server
DCS_DIR = Path(DATA_DIR, 'conllu', 'files')
# DCS_TEST = Path(DATA_DIR, 'conllu', 'tests')

DCS_JSON = Path(DATA_DIR, 'dcs_filtered.json')

class Token(object):
	__slots__ = [
		'index',        # i-j if multi-word
		'form',         # not present for compound components
		'lemma',        # not present for compound
		'upos',
		'xpos',
		'feats',
		'head',         # safely ignore
		'deprel',       # safely ignore
		'deps',         # safely ignore
		'misc',         # safely ignore
		'lemma_id',
		'unsandhied',   # unsandhied form for normal tokens
		'sem_concept',
		'multi',        # end index of mwt
	]

	def __init__(self,
				index=0,
				form=None,
				lemma=None,
				upos=None,
				xpos=None,
				feats=None,
				head=None,
				deprel=None,
				deps=None,
				misc=None,
				lemma_id=0,
				unsandhied=None,
				sem_concept=None,
				multi=0,):
		self.index=int(index)
		self.form=form
		self.lemma=None if not lemma or (lemma == "_" and upos != "PUNCT") else lemma
		self.upos=upos
		self.xpos=None if not xpos or xpos == "_" else xpos
		self.feats=None if not feats or feats == "_" else feats
		self.head=None if not head or head == "_" else int(head)
		self.deprel=deprel
		self.deps=None if not deps or deps == "_" else deps
		self.misc=None if not misc or misc == "_" else misc
		self.lemma_id=int(lemma_id)
		self.unsandhied=None if not unsandhied else unsandhied
		self.sem_concept=None if not sem_concept else sem_concept
		self.multi=int(multi)

	@classmethod
	def from_str(cls, s):
		columns = s.rstrip().split("\t")
		if "-" in columns[0]:  # multi-word tokens
			begin, end = columns[0].split("-")
			return cls(index=begin, form=columns[1], multi=end)
		else:
			return cls(*columns)

	def get_unsandhied(self):
		return self.unsandhied

	def get_multi_info(self):
		return self.index, self.multi, self.form

	def is_multi(self):
		return True if self.multi else False


class Sentence(object):

	__slots__ = ['tokens', 'comments', 'tokens_dict']

	def __init__(self, tokens=[], comments={}):
		self.tokens = tokens
		self.comments = comments

		self.tokens_dict = construct_dict(self.tokens)

	def get_sent_id(self):  # keeping reference
		return self.comments['# text_line_id'].rstrip()

	def get_text(self):
		return self.comments['# text_line'].rstrip()

	def reconstruct_unsandhied_sequence(self):
		seq = []
		j = 0
		tok_index = 1
		while j < len(self.tokens):
			t = self.tokens[j]
			if t.is_multi():
				start, end, _ = t.get_multi_info()
				span = end-start+1
				seq.append(self.reconstruct_unsandhied_compound(start, end))
				multi = tok_index + span -1 # skip components
				j = multi
				tok_index = multi + 1
			else:
				seq.append(t.get_unsandhied())
			j += 1
			tok_index += 1
		return seq

	def reconstruct_unsandhied_compound(self, start, end):
		return [self.tokens_dict[i].unsandhied for i in range(start, end+1)]


def construct_dict(tokens):
	return {t.index: t for t in tokens}

def gather_sents(conllu):
	with open(conllu, 'r', encoding='utf-8') as c:
		comments, tokens = {}, []
		for i, line in enumerate(c):
			if line.startswith('# text_line'):  # is comment
				k, _, v = line.partition(": ")
				comments[k] = v
			if re.match(r"^\d+.*$",line):
				tokens.append(Token.from_str(line))
			if line == '\n' and comments:
				sent = Sentence(tokens, comments)
				comments, tokens = {}, []
				yield sent
		sent = Sentence(tokens, comments)  # handles last sentence, connlu file is not well-formed
		yield sent

def list_dcs(dcs_dir):
	if not dcs_dir.exists():
		logger.warning(f'DCS directory {dcs_dir} does not exist.')
	else:
		return dcs_dir.rglob('*.conllu')

def create_json(dataset, filename=DCS_JSON):
	with open(filename, 'w', encoding='utf-8') as f:
		json.dump(dataset, f, ensure_ascii=False, indent=4)


def load_conllu_data(jsonfile=DCS_JSON):
	if DCS_JSON.is_file():
		with open(jsonfile, 'rb') as data:
			dataset = json.load(data)
	else:
		conllu_files_iter = list_dcs(DCS_DIR)
		# conllu_files_iter = list_dcs(DCS_TEST)

		sentences = chain.from_iterable(  # process all conllu files
			[
				gather_sents(conllu)
				for conllu in conllu_files_iter
			]
		)


		dataset = []

		for sent in sentences:
			sent_id_dcs = sent.get_sent_id()
			sent_text = sent.get_text()
			unsandhied_tokenized = sent.reconstruct_unsandhied_sequence()
			# ---- segmented sentence ----
			segments = []
			for item in unsandhied_tokenized:
				if isinstance(item, list):
					segments.extend(item)
				else:
					segments.append(item)
			segmented_sentence = ' '.join(segments)

			# ---- filter out entries with '_' (unknown values) ----
			if not '_' in segmented_sentence:

				datapoint = {
					'sent_id': sent_id_dcs,
					'joint_sentence': sent_text,
					'segmented_sentence': segmented_sentence,
					't1_ground_truth': unsandhied_tokenized
				}

				dataset.append(datapoint)

				# pp.pprint(datapoint)

		create_json(dataset)

	return dataset


if __name__ == '__main__':

	dataset = load_conllu_data(DCS_JSON)
	pprint(dataset[-2:])

