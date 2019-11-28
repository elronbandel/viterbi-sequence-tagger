from collections import deque

from MLETrain import MLE, read_counter_from_file
from old_code import data_tools
from old_code.HMMGraph import *

Record = namedtuple('Record', ['p','node', 'source'])

class ViterbiTagger:
    def __init__(self, mle, n_gram = 3):
        self.mle = mle
        self.n_gram = n_gram
        self.n_gram_offset = n_gram - 1

    def tag(self, line):
        g = HMMGraph(self.mle, self.n_gram)
        node = g.start_node(line)
        records = [dict() for _ in range(len(line) - self.n_gram_offset)]
        for child in g.transitions(node):
            records[child.time_stamp][g.node_tag(child)] = Record(g.transition_probability(node, child), node, None)
        for time_stamp in range(len(records)):
            for tag, record in records[time_stamp].items():
                node = record.node
                for child in g.transitions(node):
                    p = g.transition_probability(node, child) + record.p
                    c_tag = g.node_tag(child)
                    if c_tag not in records[child.time_stamp]:
                        records[child.time_stamp][c_tag] = Record(p, child, tag)
                    elif records[child.time_stamp][c_tag].p <= p:
                        records[child.time_stamp][c_tag] = Record(p, child, tag)


        tag = max(records[-1].items(), key=lambda record: record[1].p)[0]
        tags = deque()
        for i in reversed(range(len(records))):
            tags.appendleft(tag)
            tag = records[i][tag].source
        return [line[0]] * self.n_gram_offset + list(tags)


def test():
    q_counter = read_counter_from_file("q.mle")
    e_counter = read_counter_from_file("e.mle")
    mle = MLE(q_counter, e_counter)
    line = "he walked home quickly".split()
    line_with_start_sym = data_tools.add_start_symbol_to_list_of_words(line, n_gram=3)
    tagger = ViterbiTagger(mle)
    print(tagger.tag(line_with_start_sym))




if __name__ == '__main__':
    test()