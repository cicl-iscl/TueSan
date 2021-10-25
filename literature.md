
# Literature

## Word Segmentation

- [S. Krishnan and A. Kulkarni, 2019] [Sanskrit Segmentation Revisited](https://aclanthology.org/2019.icon-1.12.pdf)
  - proposes modifications for the Heritage Segmenter (FST that produces all possible segmentations): 
  End goal is segmentation, ignore phases except for those that are related to compounds, calculates a confidence value C for each segmentation solution as the POP (product-of-products) of the word and transition probabilities. 
  - Section 2 gives a brief review of exsisting approaches to the segmentation problem, it serves as an entrypoint to other papers
    - [Huet, 2003] Heritage Segmenter - FST, one for external (sentence) sandhi and one for internal (word) sandhi
    - [Mittal, 2010] Optimality Theory to validate possible segmentations produced by rule-based FST.
    - [Kumar et al., 2010] compound segmentation using probabilistic method
    - [Natarajan and Charniak, 2011] Bayesian segmentation algorithm
    - [Krishna et al., 2016] Path Constrained Random Walk framework for selecting nodes from the segmentation graph with possible solutions, combining morphological features and word co-occurence features from a manually tagged corpus (Hellwig, 2009).
    - [Aralikatte et al., 2018] Double Decoder RNN with attention as seq2(seq)^2: one decoder learns the split locations (where), the other learns the splits themselves (how).
    - [Hellwig and Nehrdich, 2018] Character-level Recurrent and Convolutional Neural Networks, disregard the difference between splitting joint words and compounds, does not require feature engineering or external linguistic resources, claims to work well with just the parallel versions of raw and segmented text.
    - [Krishna et al., 2018] a structured prediction framework using EBM (energy based model), solves word segmentation AND morphological tagging tasks, graph-based parsing techniques. 

- [A. S. R and A. Kulkarni, 2014] [Segmentation of Navya-Nya ̄ya Expressions](https://aclanthology.org/W14-5141/)
- [Krishna et al., 2016] [Word Segmentation in Sanskrit Using Path Constrained Random Walks](https://aclanthology.org/C16-1048/)
- [Aralikatte et al., 2018] [Sanskrit Sandhi Splitting using seq2(seq)^2](https://aclanthology.org/D18-1530/)
- [Hellwig and Nehrdich, 2018] [Sanskrit Word Segmentation Using Character-level Recurrent and Convolutional Neural Networks](https://aclanthology.org/D18-1295/)
- [Krishna et al., 2018] [Free as in Free Word Order: An Energy Based Model for Word Segmentation and Morphological Tagging in Sanskrit](https://aclanthology.org/D18-1276/)
- [S. Dave et al., 2021] [Neural Compound-Word (Sandhi) Generation and Splitting in Sanskrit Language](https://dl.acm.org/doi/10.1145/3430984.3431025)

## Other
- [A. Kulkarni, 2013] [A Deterministic Dependency Parser with Dynamic Programming for Sanskrit](https://aclanthology.org/W13-3718/)
- [Krishna et al., 2020] [A Graph-Based Framework for Structured Prediction Tasks in Sanskrit](https://aclanthology.org/2020.cl-4.4/)
- [G. Inoue et al., 2017] [Joint Prediction of Morphosyntactic Categories for Fine-Grained Arabic Part-of-Speech Tagging Exploiting Tag Dictionary Information](https://aclanthology.org/K17-1042/)
