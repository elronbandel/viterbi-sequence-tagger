from collections import namedtuple

HMMNode = namedtuple("HMMNode", ["time_stamp", "sequence", "n_tags_history"])

class HMMGraph:
    def __init__(self, mle, n_gram):
        self.mle = mle
        self.n_gram = n_gram
        self.n_gram_offset = n_gram - 1

    def start_node(self, sequence):
        return HMMNode(-1, sequence, [sequence[0]] * self.n_gram)

    def transitions(self, node):
        if node.time_stamp + self.n_gram >= len(node.sequence):
            return []
        return (HMMNode(node.time_stamp + 1, node.sequence, node.n_tags_history[1:] + [tag]) for tag in self.mle.tags)

    def transition_probability(self, node1, node2):
        p = self.mle.getE(node2.sequence[node2.time_stamp + self.n_gram_offset], node2.n_tags_history[-1]) * self.mle.getQ(node2.n_tags_history)
        return p
    @staticmethod
    def node_tag(node):
        return node.n_tags_history[-1]

