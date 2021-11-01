# Changelog

To keep us posted of what's been changed and added ;-), and to remind the forgetful self. 

## 2021-10-31
----
### Added
- `conllu_parser.py`: parses DCS conllu files, turns out "bad data" comes from incomplete annotations...

### Changed
- `prepare_data.py`: tries to create a similar processed dataset, without 'allowed_words' since we have no graphml data for other sents.


## 2021-10-30
----
### Added
- `train_dataset.pickle`: generated using Leander's code for the full training dataset (90 000), saved as pickle for faster loading. Did **NOT** remove punctuation!
- `logger.py`: configure logger
- python files which simply move Leander's code from Colab notebooks around
	- `utils.py`: utilities for accessing data, provided with the task; a helper method to peek into a specific sentence.
	- `prepare_data.py`: prepares the (training) dataset
	- `reconstruct.py`: try to reconstruct unsandhied tokens, error analysis
	- `task1.py`: model
- description of data folder structure in `README.md`
- `pyproject.toml` for dependency management (and virtual environment) with poetry
- `CHANGELOG.md` to keep the changes visible for everyone.

### Changed
- `reconstruct.py`: 
	- added `'` as a vowel, edit distance to `a` is still 1.
	- more error types to analyse, added `next token` for easier inspection
- `literature.md` and `meetings.md`: add paper for the Professor Forcing Algorithm
 
