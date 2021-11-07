import json
import pickle

from helpers import load_data, load_sankrit_dictionary, check_specific_sent
from vocabulary import make_vocabulary
from generate_train_data import construct_dataset
from dataset import index_dataset, collate_fn

from pathlib import Path
from logger import logger
import pprint
pp = pprint.PrettyPrinter(indent=4)


if __name__ == '__main__':
    with open("config.cfg") as cfg:
        config = json.load(cfg)

    translit = config['translit']

    # Load data, data is a dictionary indexed by sent_id
    logger.info("Load data")
    train_data = load_data(config['train_path'], translit)
    eval_data = load_data(config['eval_path'], translit)

    # Display an example sentence
    sent = check_specific_sent(eval_data, 582923)
    pp.pprint(sent)

    # Make vocabulary
    # whitespaces are translated to '_' and treated as a normal character
    logger.info("Make vocab")
    vocabulary, char2index, index2char, char2uni = \
        make_vocabulary(train_data.values())

    logger.debug(f"{len(vocabulary)} chars in vocab:\n{vocabulary}")

    # # Double check vocab of dev
    # eval_vocabulary, _, _, _ = \
    #   make_vocabulary(eval_data.values())
    # for char in eval_vocabulary:
    #   if char not in vocabulary:
    #       logger.warning(f"{char} not in train vocab.")

    # Construct train data
    dat = {}
    with open(config['train_path'], encoding='utf-8') as train_json:
        dat = json.load(train_json)
    train_dataset = construct_dataset(dat, translit, config['train_graphml'])
    with open(Path(config['out_folder'], 'translit_train.pickle'), 'wb') as outf:
        pickle.dump(train_dataset, outf)

    # Construct eval data
    eval_dat = {}
    with open(config['eval_path'], encoding='utf-8') as eval_json:
        eval_dat = json.load(eval_json)
    eval_dataset = construct_dataset(eval_dat, translit, config['eval_graphml'])
    with open(Path(config['out_folder'], 'translit_dev.pickle'), 'wb') as outf:
        pickle.dump(eval_dataset, outf)

    # Display an example datapoint
    pp.pprint(train_dataset[0])

    # Index data
    logger.info("Index train data")
    train_data_indexed = index_dataset(train_dataset, char2index)
    eval_data_indexed = index_dataset(eval_dataset, char2index)

    # Display an example
    pp.pprint(eval_dataset[0])
    pp.pprint(eval_data_indexed[0])
    assert len(eval_dataset[0]['sandhied_merged']) == eval_data_indexed[0][0].size(dim=0)

