"""Utilities for parsing dcs conllu files
"""
from logger import logger

def load_conll(file):
    pass

def get_mwt():
    pass

def get_nested():
    pass

def construct_json():
    pass

class Node(object):
    """
    """
    __slots__ = [
        'index',
        'form',
        'lemma',
        'upos',
        'xpos',
        'feats',
        'head',
        'deprel',
        'deps',
        'misc',
        'lemma_id',
        'unsandhied',
        'semantic_concept',
        'multi',
    ]

    def __init__(self,
                index=0,
                form=None,
                lemma=None,
                upos=None,
                xpos=None,
                feats=None,
                head=None,
                deprel=None,
                deps=None,
                misc=None,
                lemma_id=0,
                unsandhied=None,
                semantic_concept=None,
                multi=0,):
        self.index=int(index)
        self.form=form
        self.lemma=None if not lemma or (lemma == "_" and upos != "PUNCT") else lemma
        self.upos=upos
        self.xpos=None if not xpos or xpos == "_" else xpos
        self.feats=None if not feats or feats == "_" else feats
        self.head=None if not head or head == "_" else int(head)
        self.deprel=deprel
        self.deps=None if not deps or deps == "_" else deps
        self.misc=None if not misc or misc == "_" else misc
        self.lemma_id=int(lemma_id)
        self.unsandhied=None
        self.semantic_concept=None
        self.multi=int(multi)

    @classmethod
    def from_str(cls, s):
        columns = s.rstrip().split("\t")
        if "-" in columns[0]:
            begin, end = columns[0].split("-")
            return cls(index=begin, form=columns[1], misc=columns[9], multi=end)
        else:
            return cls(*columns)

    def __str__(self):
        fields = [str(getattr(self, x)) for x in self.__slots__[1:12]]
        idx = str(self.index)
        if self.multi:
            idx = "-".join((str(self.index), str(self.multi)))
        return ("\t".join([idx] + fields)).replace("None", "_").replace("[]", "_")

class Sentence(object):
    """
    """
    __slots__ = [
        'nodes',
        'multi',
        'comment',
    ]

    def __init__(self, instr=None, stream=None):
        self.nodes = [Node(index=0)]  # initialize with the dummy root node
        self.multi = dict()
        self.comment = []

        if stream:
            inp_str = self.read_sentence(stream)
        self.from_str(inp_str)

    def from_str(self, lines):
        for line in lines.splitlines():
            if line.startswith("#"):
                assert (
                    len(self.nodes) == 1
                ), "Comments are allowed only at the beginning"
                self.comment.append(line)
            else:
                node = Node.from_str(line)
                if node.multi:
                    self.multi[node.index] = node
                elif node.empty:
                    e = self.empty.get(node.index, [])
                    e.append(node)
                    self.empty[node.index] = e
                else:
                    self.nodes.append(node)

    def read_sentence(self, stream):
        """Read a sentence form a CoNLL-U file from a stream.
        The returned list of strings includes pre-sentence comment(s).
        The final empty line is read, but not added to the return value.
        """
        lines = ""
        line = stream.readline()
        while line and not line.isspace():
            lines += line
            line = stream.readline()
        return lines

