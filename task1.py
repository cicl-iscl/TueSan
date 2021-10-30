#!/usr/bin/env python3
import torch

from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from torch.optim import AdamW
from torch.nn import BCELoss

from pathlib import Path
from tqdm import tqdm
from logger import logger

from prepare_data import load_data


TRAIN_PICKLE = Path('train_dataset.pickle')


def pad(batch):
	inputs, labels = zip(*batch)
	max_len = max([len(seq) for seq in inputs])

	padded_batch = []
	for seq, labels in batch:
		padding_length = max_len - len(seq)
		seq_padding = seq.new_zeros(padding_length)
		padded_seq = torch.cat([seq, seq_padding], dim=-1)
		label_padding = labels.new_zeros(padding_length)
		padded_labels = torch.cat([labels, label_padding], dim=-1)

		padded_batch.append((padded_seq, padded_labels))

	return padded_batch


def collate_fn(batch):
	padded_batch = pad(batch)
	padded_inputs, padded_labels = zip(*padded_batch)

	inputs = torch.stack(padded_inputs)
	labels = torch.stack(padded_labels)

	return inputs, labels


class SegmenterModel(nn.Module):
	def __init__(self, vocabulary_size, embedding_dim, hidden_dim):
		super().__init__()

		self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
		self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
						   bidirectional=True, batch_first=True)
		self.projection = nn.Sequential(
			nn.Linear(2*hidden_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 32),
			nn.ReLU(),
			nn.Linear(32, 1),
			nn.Sigmoid()
		)

	def _sequence_mask(self, lengths):
		# from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
		batch_size = lengths.shape[0]
		max_len = torch.max(lengths).item()
		seq_range = torch.arange(0, max_len).long()
		seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
		seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
		return seq_range_expand < seq_length_expand

	def forward(self, input):
		batch_size = input.shape[0]
		lengths = torch.sum(input != 0, dim=-1).long()
		mask = self._sequence_mask(lengths)

		input = self.embedding(input)
		input = pack_padded_sequence(input, lengths, batch_first=True,
									 enforce_sorted=False)
		input, _ = self.rnn(input)
		input, lengths = pad_packed_sequence(input, batch_first=True)
		predictions = self.projection(input)
		predictions = predictions.reshape(batch_size, -1)
		masked_predictions = predictions * mask

		return masked_predictions


if __name__ == '__main__':

	# ---- load dataset ----
	task1_dataset = load_data(TRAIN_PICKLE)
	dataset = task1_dataset  # rename


	vocabulary = set()
	for dp in dataset:
		text = dp['sandhied_merged']
		# note to self: 'sandhied_merged', constructed by 'make_labels()', disregard all space characters
		vocabulary.update(set(text))

	specials = ["<PAD>", "<UNK>"]
	vocabulary = specials + list(sorted(vocabulary))

	char2index = {char: index for index, char in enumerate(vocabulary)}
	index2char = {index: char for char, index in char2index.items()}


	indexed_dataset = []
	for dp in dataset:
		text, labels = dp['sandhied_merged'], dp['labels']
		indexed_text = [char2index[char] for char in text]
		indexed_text = torch.LongTensor(indexed_text)
		labels = torch.from_numpy(labels).float()
		indexed_dataset.append([indexed_text, labels])

	# ---- test print vocab ----
	logger.info(f"Vocab contains {len(vocabulary)} items")
	logger.info(vocabulary)  # "'" has been added to vocab


	batch_size = 64
	epochs = 1

	dataloader = DataLoader(indexed_dataset, batch_size=batch_size,
							collate_fn=collate_fn)
	model = SegmenterModel(len(vocabulary), 16, 32)
	optimizer = AdamW(model.parameters())
	criterion = BCELoss()

	running_loss = None
	for epoch in range(epochs):
		batches = tqdm(dataloader)
		for inputs, labels in batches:
			y_pred = model(inputs)

			optimizer.zero_grad()
			loss = criterion(y_pred, labels)
			loss.backward()
			optimizer.step()

			detached_loss = loss.detach().cpu().item()
			if running_loss is None:
				running_loss = detached_loss
			else:
				running_loss = 0.95 * running_loss + 0.05 * detached_loss

			batches.set_postfix_str(
				"Loss: {:.2f}, Running Loss: {:.2f}" \
				.format(detached_loss * 100, running_loss * 100)
				)

	dataloader = DataLoader(indexed_dataset, batch_size=4,
							collate_fn=collate_fn)

	with torch.no_grad():
		inputs, labels = next(iter(dataloader))
		y_pred = model(inputs).detach().cpu()
		y_pred = torch.round(y_pred)

		logger.info(f'y_pred:\t{y_pred}')
		logger.info(f'labels:\t{labels}')
