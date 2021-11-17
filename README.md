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
#### T1 Segmentation
----
Trained for 20 epochs, using current hyperparameters in `config.cfg`.
```
Duration: 2295.23 secs
Scores
----
task_1_precision: 36.62
task_1_recall: 42.17
task_1_f1score: 39.20
task_1_tscore: 42.17
```
- Example wrong prediction:
```
Source:	['d', 'r', 'a', 'v', 'y', 'A', 'B', 'A', 'v', 'e', ' ', 'd', 'v', 'i', 'j', 'A', 'B', 'A', 'v', 'e', ' ', 'p', 'r', 'a', 'v', 'A', 's', 'e', ' ', 'p', 'u', 't', 'r', 'a', 'j', 'a', 'n', 'm', 'a', 'n', 'i']
Gold:	['d', 'r', 'a', 'v', 'y', 'a', ' ', 'a', 'B', 'A', 'v', 'e', ' ', 'd', 'v', 'i', 'j', 'a', ' ', 'a', 'B', 'A', 'v', 'e', ' ', 'p', 'r', 'a', 'v', 'A', 's', 'e', ' ', 'p', 'u', 't', 'r', 'a', ' ', 'j', 'a', 'n', 'm', 'a', 'n', 'i']
Pred:	['d', 'r', 'a', 'v', 'ya ', 'a a', 'B', 'A', 'v', 'e', ' ', 'd', 'v', 'i', ' ', 'ja ', 'a a', 'B', 'A', 'v', 'e', ' ', 'p', 'r', 'a', 'v', 'A', 's', 'e', ' ', 'p', 'u', 't', 'r', 'a', ' ', 'j', 'a', 'n', 'm', 'a', 'n', 'i']
Rules:	[('d', '<COPY>'), ('r', '<COPY>'), ('a', '<COPY>'), ('v', '<COPY>'), ('y', 'ya '), ('A', 'a a'), ('B', '<COPY>'), ('A', '<COPY>'), ('v', '<COPY>'), ('e', '<COPY>'), (' ', '<COPY>'), ('d', '<COPY>'), ('v', '<COPY>'), ('i', '<INSERT SPACE>'), ('j', 'ja '), ('A', 'a a'), ('B', '<COPY>'), ('A', '<COPY>'), ('v', '<COPY>'), ('e', '<COPY>'), (' ', '<COPY>'), ('p', '<COPY>'), ('r', '<COPY>'), ('a', '<COPY>'), ('v', '<COPY>'), ('A', '<COPY>'), ('s', '<COPY>'), ('e', '<COPY>'), (' ', '<COPY>'), ('p', '<COPY>'), ('u', '<COPY>'), ('t', '<COPY>'), ('r', '<COPY>'), ('a', '<INSERT SPACE>'), ('j', '<COPY>'), ('a', '<COPY>'), ('n', '<COPY>'), ('m', '<COPY>'), ('a', '<COPY>'), ('n', '<COPY>'), ('i', '<COPY>')]
S:	dravyABAve dvijABAve pravAse putrajanmani
G:	dravya aBAve dvija aBAve pravAse putra janmani
P:	dravya a aBAve dvi ja a aBAve pravAse putra janmani
```

- Observations
    - **Over application of `<INSERT SPACE>` rule**, resulting in a number of single characters, one simple thing to do is maybe postprocess the predictions to (randomly) merge these characters with their predecessors and successors.
        - For forms like `dravya a aBAve` and `ja a aBAve`, choose to ignore the middle `a`?  

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

