""" Convert between IAST and internal representation used which is similar to SLP1.
	Taken from: https://github.com/OliverHellwig/sanskrit/blob/master/papers/2018emnlp/code/data_loader.py

"""

UNICODE2INTERN = [
    (u"ā", "A"),
    (u"ī", "I"),
    (u"ū", "U"),
    (u"ṛ", "R"),
    (u"ṝ", "L"),  # ??
    (u"ḷ", "?"),
    (u"ḹ", "?"),
    (u"ai", "E"),
    (u"au", "O"),
    # gutturals
    (u"kh", "K"),
    (u"gh", "G"),
    (u"ṅ", "F"),
    # palatals
    (u"ch", "C"),
    (u"jh", "J"),
    (u"ñ", "Q"),
    # retroflexes
    (u"ṭh", "W"),
    (u"ṭ", "w"),
    (u"ḍh", "X"),
    (u"ḍ", "x"),
    (u"ṇ", "N"),
    # dentals
    (u"th", "T"),
    (u"dh", "D"),
    # labials
    (u"ph", "P"),
    (u"bh", "B"),
    # others
    (u"ś", "S"),
    (u"ṣ", "z"),
    (u"ṃ", "M"),
    (u"ḥ", "H"),
]


def unicode_to_internal_transliteration(s):
    """
    Transforms from IAST to the internal transliteration
    """
    for src, dst in UNICODE2INTERN:
        s = s.replace(src, dst)
    return s


def internal_transliteration_to_unicode(s):
    for src, dst in UNICODE2INTERN:
        s = s.replace(dst, src)
    return s
