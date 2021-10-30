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
      ├── graphml_dev
      |   ├── ddd.graphml
      │   └── ...
      ├── final_graphml_train
      |   ├── ddd.graphml
      |   └── ...  
      ├── wsmp_train.json
      └── wsmp_dev.json
```
Data can be accessed from `rummelhart` server, `/data/jingwen/sanskrit/`.
