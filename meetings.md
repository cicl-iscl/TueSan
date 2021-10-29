# 1029
----
### Sandhi rules grouped by reverse operation
| Operation        |Source         | Target | Description |
| -----------------|:-------------:| :------:|:-----------|
|||||
|**vowels**||||
|||||
|resolve long vowel|-ā/ī/ū-|-a/i/u + a/i/u-|simple vowel with simple vowel of the same kind (long/short)|
|resolve long vowel|-ā/ī/ū-|-ā/ī/ū + ā/ī/ū-|simple vowel with simple vowel of the same kind (long/short)|
|resolve long vowel|-ā/ī/ū-|-a/i/u + ā/ī/ū-|simple vowel with simple vowel of the same kind (long/short)|
|resolve long vowel|-ā/ī/ū-|-ā/ī/ū + a/i/u-|simple vowel with simple vowel of the same kind (long/short)|
|||||
|resolve long vowel exception|amī, ī(Interjection), als Dualendung, "jene"|keep separate||
|||||
|resolve Hochstufe|-e/o/ar/al-|-a/ā + i/u/ṛ/ḷ-|a/ā simple vowel with simple vowel of other kinds (long/short)-->eine Stufe hoch|
|resolve Dehnstufe|-ai/au/ār/āl-|-a/ā + e/o/ar/al-|a/ā simple vowel with dipthongs of other kinds-->zwei Stufen hoch|
|||||
|resolve Halbvokal(i-kind)|-y-|-i/ī + non-i-kind-|non-a/ā simple vowel with vowel of different kind -> non-a/ā simple vowel changes to corresponding Halbvokal|
|resolve Halbvokal(u-kind)|-v-|-u/ū + non-u-kind-|non-a/ā simple vowel with vowel of different kind -> non-a/ā simple vowel changes to corresponding Halbvokal|
|resolve Halbvokal(ṛ-kind)|-r-|-ṛ/ṝ + non-ṛ-kind-|non-a/ā simple vowel with vowel of different kind -> non-a/ā simple vowel changes to corresponding Halbvokal|
|resolve Halbvokal(ḷ-kind)|-l-|-ḷ + non-ḷ-kind-|non-a/ā simple vowel with vowel of different kind -> non-a/ā simple vowel changes to corresponding Halbvokal|
|||||
|resolve Avagraha|-'-|-e/o + a-|te + api --> te'pi|
|||||
|resolve complex a|-a non-a-vowel-|-e/o + non-a-vowel(including ā)-|vane + āste --> vana āste|
|resolve complex ā|-ā vowel-|-ai + vowel-|tasmai + adāt --> tasmā adāt|
|resolve complex āv|-āvvowel-|-au + vowel-|tau + ubhau  -> tāvubhau|
|||||
|**consonants**||||
|||||
|keep visarga|-ḥ unvoiced V/L/Z|-ḥ unvoiced V/L/Z|ḥ remain unchanged before unvoiced velars (k, kh), unvoiced labials (p, ph) and Zischlaute (ṣ, ś, s)|
|visarga changes|-ś/ṣ/s-|-ḥ unvoiced P/R/D|ḥ changes before unvoiced palatal (ca, cha), unvoiced retroflex (ṭ, ṭh) , unvoiced dental (t, th) into corresponding Zischlaute (ś, ṣ, s)|
|visarga changes|-r-|-<non-a/ā-vowel>ḥ voiced(except r-)||
|!|-<non-a/ā-vowel> elongated-r-|-<non-a/ā-vowel>ḥ r-||
|||||
||to be continued|||
|||||
|||||

### Pausaform
Max. 1 consonant before space/end of sentence -- "Pausa", with exceptions.
Types of consonants allowed:

- 1.Group: k, ṅ **velar**
- 3.Group: ṭ, ṇ **retroflex**
- 4.Group: t, n **dental**
- 5.Group: p, m **labial**
- Visarga: ḥ
- y, l, v

Other consonants have to transform to one of the above allowed consonants before a pause. 

2. Group **palatal** changes to 1.Group: k, ṅ **velar** respectively (Exception: sometimes j -> ṭ).

ṣ, h --> ṭ or sometimes k.

ś --> k or sometimes ṭ

### External Data
Possible data augmentation:

[DCS in conllu format](https://github.com/OliverHellwig/sanskrit/tree/master/dcs/data/conllu): Oliver Hellwig: Digital Corpus of Sanskrit (DCS). 2010-2021.

- The train/dev data is also taken from DCS, so I guess we can't use it.
- I was trying to figure out if the sent_ids are the same, turns out they are not. Can't be used to retrieve correct segmentations.
- Still, the conllu data can be of some use, although the format is not strictly CoNLL-U, 13 fields instead of 10, had to write a simple parser.

[Sanskrit UD Treebank](https://github.com/UniversalDependencies/UD_Sanskrit-UFAL):

- a very small corpus ~230 sentences, though the text is of similar style. Lemma and form retrievable.
- could use this as test

#### Graphml

- *color_class*: rough approx. of word category
- morph: groundtruth analysis
- *sense*: polysemy
- stem: groundtruth stem
- word: word form


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
 - ~~What we actually need to disambiguate the non-deterministic solutions is ALL the *positions* where sandhi occurs.~~ But can we solve it this way? How is it different from knowing `A` in `vidyApyata` should be split?? I guess morphological info is still needed, e.g. agreement of number, gender, etc.
 - How do we learn the positions?
 - We can see a joint chunk/sentence as consecutive fuzzy words with head and tails waiting for specification. This would require we have some kind of lexicon and recognise the fuzzy words before we do anything else. This is essentially what the Double Decoder RNN was doing, learning the positions of splits.
 - We should also have an acoustic model that either minimise the effort of pronouncing a joint word (define objective), or maximise the conditional probability of the output segment given the input sentence.
 - Refer to the paper mentioned by Hellwig, *Segmental RNNs for acoustic modelling*.



 
 
