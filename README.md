# TüSan
Word Segmentation and Morphological Parsing for Sanskrit

### Prerequisites

Python Version >= 3.8, <3.11 (for numpy)

This project supports using [poetry](https://python-poetry.org/) for dependency management. Follow these [instructions](https://python-poetry.org/docs/#installation) to install poetry.
```
cd TueSan
poetry install
```

To activate virtual environment, run
```
poetry shell
```

### Data Folder Structure
----
```
    ./sanskrit/
      ├── graphml_dev                     # auxiliary graphml data
      |   ├── ddd.graphml
      │   └── ...
      ├── final_graphml_train
      |   ├── ddd.graphml
      |   └── ...  
      ├── conllu                          # DCS, from https://github.com/OliverHellwig/sanskrit/tree/master/dcs/data/conllu
      |   ├── lookup
      |   |   ├── dictionary.csv
      |   |   ├── pos.csv
      |   |   └── word-senses.csv  
      |   └── files
      |       ├── <subfolders>
      |       |   ├── xxx.conllu
      |       |   └── ...
      |       ├── xxx.conllu
      |       └── ...  
      ├── dcs_filtered.json               # DCS for task 1, sentences with incomplete annotations are filtered out
      ├── dcs_processed.pickle            # DCS for task 1, with 'sandhied_merged', 'labels', etc.
      ├── wsmp_train.json                 # primary data
      └── wsmp_dev.json
      
```
Data can be accessed from the server, `/data/jingwen/sanskrit/`.


### To-Do
----
- hyperparameter tuning T3 > T1, ray.tune?

