# 1027
----

## Things to discuss
- Which task(s) do we want to work on? Does any of us have any ideas how to proceed?
- Most importantly, we should set a timeline (maybe after our first meeting with Cagri)
- What's included in the dataset?
  - primary data: JSON files
  - auxiliary data: Graphml files--optional use, graph representing all possible segmentations
  - utilities for reading the JSON and Graphml files
- Do we need any form of external linguistic knowledge of Sanskrit (besides understanding the problem at hand)? 
  - e.g. a complete list of sandhi rules
    - grouping of consonants (dentals, labials, etc./voiced vs. unvoiced/aspirated vs. unaspirated)
    - vowel quality (simple vs. dipthong, short vs. long)...
    - visarga, anusvara, avagraha...
  - most probably we won't need them
 
 ### Reformulate the (Segmentation) Problem
 
 - Euphonic assimilations obscure word boundaries in speech. 
 - Sanskrit has an oral tradition long before the orthographic sytem was invented, and the writings faithfully capture the changes:
     - `vasati atra  -> vasatyastra`, `harit dandam -> hariddandam` [change on end of first word]
     - `vidyA Apyata -> vidyApyata` (A represents long vowel a, usually written as a with a bar above) [shared]
     - `blahblaha ablah -> blahblahAblah` [meeting of two short vowels of the same type -> a shared long vowel]
     - `-a u- -> -o-` [changes vowel quality]
     - no changes but words are written together [no space despite no sandhi applied]
     - `kutah + Agacchasi -> kuta Agacchasi` an example case of dropping final visarga `:` (usually written as h with a dot under), but words are written separately [space-separated despite a sandhi rule is applied]
     - `tat + hi -> taddhi` [`-t + h- -> -d + dh-`] [changes both, changes consonant quality]
     - ......
 - Determining word boundaries is not a trivial task.
 - *Can it be considered as a speech to text problem? Or use algorithms for speech recognition?*
 
 - Let *`P`* be the set of phonemes, and *`R`* the set of rules. And we define a mapping *`s: P x P -> P`*. 
 - The mapping `s` (sandhi) can be seen as "choosing to apply a rule `r` in *`R`* given `word-final p1` and `word-initial p2` to get `p3`", since the application of sandhi rules is deterministic.
 - *Can we reformulate the euphonic assimilation process as a transformation that acts on the embeddings of `p1` and `p2` and try to solve for `s` given the embedding of `p3`?*
 - character-based or syllable-based?
 - But the inverse is non-deterministic, for example, the long vowel `A` can be separated in a least two ways: `-a + a-` and `-A + A-`.
 - It seems that we need extra information to make a decision! What do we need?
 - It seem that we should know which words to join, but we don't actually need the morphological/lexical information of the words, because the euphonic process operates on its own, the only basis is the proximity of sounds.
 - What we actually need to disambiguate the non-deterministic solutions is the *position* where sandhi occurs.
 - How do we learn the positions?
 - We can see a joint chunk/sentence as consecutive fuzzy words with head and tails waiting for specification. This would require we have some kind of lexicon and recognise the fuzzy words before we do anything else. This is essentially what the Double Decoder RNN was doing, learning the positions of splits.
 - We should also have an acoustic model that either minimise the effort of pronouncing a joint word (define objective), or maximise the conditional probability of the output segment given the input sentence.
 - Refer to the paper mentioned by Hellwig, *Segmental RNNs for acoustic modelling*.
 
 
