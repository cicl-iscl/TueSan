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

### Error Analyses
----
#### T2 RuleClassification
- Example wrong prediction:  
```
Pred: iic.	Aha
True: pft. ac. sg. 3	ah
CandRules: [('', '', '', 'f. sg. nom.'), ('', '', '', 'iic.'), ('', '', '', 'm. sg. voc.'), ('', '', '', 'adv.'), ('', '', '', 'n. sg. acc.'), ('', 'a', '', 'ca. imp. ac. sg. 2'), ('', 'a', '', 'imp. [1] ac. sg. 2'), ('', '', '', 'm. sg. nom.'), ('', '', '', 'n. sg. nom.'), ('', '', 'n', 'iic.'), ('', '', 'n', 'n. sg. acc.'), ('', '', 'n', 'n. sg. nom.'), ('', '', '', 'inf.'), ('', '', '', 'ind.'), ('', '', '', 'abs.'), ('', '', '', 'prep.'), ('', '', '', 'part.'), ('', '', '', 'conj.'), ('', 'a', '', 'imp. [10] ac. sg. 2')]
PredRule: ('', '', '', 'iic.')
TrueRule: <Other>
Not in candidates.
```
- Observations
    - fails to generate rules in the first place (especially for **extremely short roots** like `ah`, `BU`, `kR`).
    - doesn't deal with irregular stems (from present stem `gacCa` cannot recover `gam`)
    - doesn't deal with idiosyncratic forms of `yad`, `tad`, `idam`, for example model prediction for `saH` is:
    ```
    Pred: n. sg. acc.	sas
    True: m. sg. nom.	tad
    CandRules: [('', '', '', 'f. sg. nom.'), ('', 'H', '', 'f. pl. nom.'), ('', 'H', '', 'm. sg. nom.'), ('', '', '', 'iic.'), ('', '', '', 'm. sg. voc.'), ('', 'H', '', 'm. pl. nom.'), ('', 'aH', 'A', 'm. sg. nom.'), ('', 'H', '', 'f. pl. acc.'), ('', 'aH', '', 'm. pl. acc.'), ('', 'aH', '', 'm. pl. nom.'), ('', '', '', 'adv.'), ('', '', '', 'n. sg. acc.'), ('', 'aH', '', 'm. sg. g.'), ('', '', '', 'm. sg. nom.'), ('', '', '', 'n. sg. nom.'), ('', 'H', '', 'f. sg. nom.'), ('', '', 'n', 'iic.'), ('', '', 'n', 'n. sg. acc.'), ('', '', 'n', 'n. sg. nom.'), ('', 'H', 's', 'n. sg. acc.'), ('', 'H', 's', 'n. sg. nom.'), ('', 'aH', '', 'n. sg. g.'), ('', 'aH', '', 'f. pl. nom.'), ('', 'aH', '', 'f. sg. g.'), ('', '', '', 'inf.'), ('', 'aH', '', 'm. sg. abl.'), ('', '', '', 'ind.'), ('', 'H', 's', 'iic.'), ('', 'H', 's', 'm. sg. nom.'), ('', 'H', 's', 'adv.'), ('', 'aH', '', 'f. pl. acc.'), ('', '', '', 'abs.'), ('', '', '', 'prep.'), ('', 'aH', '', 'n. sg. abl.'), ('', '', '', 'part.'), ('', 'aH', '', 'f. sg. abl.'), ('', '', '', 'conj.')]
    PredRule: ('', 'H', 's', 'n. sg. acc.')
    TrueRule: <Other>
    Not in candidates.
    ```
    - tag prediction often misses `gender` > `case`, `number` is more or less ok.

#### T2 Seq2Seq-Decoding
 To follow.


### To-Do
----
- look at T3 rule generation
- check T1

