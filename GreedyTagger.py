from MLETrain import MLE, read_counter_from_file
from old_code import data_tools
from HMMGraph import *

#g

class GreedyTagger:
    def __init__(self, mle, n_gram = 3):
        self.mle = mle
        self.n_gram = n_gram
        self.n_gram_offset = n_gram - 1

    def tag(self, line):
        g = HMMGraph(self.mle, self.n_gram)
        node = g.start_node(line)
        tags = [line[0]] * (self.n_gram_offset - 1)
        while node:
            tags.append(g.node_tag(node))
            p_max, next = 0, None
            for child in g.transitions(node):
                p = g.transition_probability(node, child)
                if p_max <= p:
                    p_max, next = p, child
            node = next
        return tags


def test():
    q_counter = read_counter_from_file("q.mle")
    e_counter = read_counter_from_file("e.mle")
    mle = MLE(q_counter, e_counter)
    line = "he walked home quickly".split()
    line_with_start_sym = data_tools.add_start_symbol_to_list_of_words(line, n_gram=3)
    tagger = GreedyTagger(mle)
    print(tagger.tag(line_with_start_sym))




if __name__ == '__main__':
    test()