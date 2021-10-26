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
 
 - Let `*P*` be the set of phonemes, and `*R*` the set of rules. And we define a mapping `*s*: *P* x *P* -> *P*`. 
 - The mapping `s` (sandhi) can be seen as "choosing to apply a rule `r` in `R` given `word-final *p1*` and `word-initial *p2*` to get `*p3*`", since the application of sandhi rules is deterministic.
 
 
