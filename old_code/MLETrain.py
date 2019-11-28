import numpy as np

from DataTools import DataProcessor, DataLoader
import DataTools
from collections import Counter


def concat(words):
    return ' '.join(words)

def tag(word_tag_string):
    return word_tag_string.split()[1]

def count_emle(data):
    counter = Counter()
    for line in data:
        for word_tag in line:
            counter[concat(word_tag)] += 1
    return counter


def count_qmle(data):
    counter = Counter()
    for line in data:
        tags = DataTools.tags(line)
        for pair in zip(tags, tags[1:]):
            counter[concat(pair)] += 1
        for triplet in zip(tags, tags[1:], tags[2:]):
            counter[concat(triplet)] +=1
    return counter


def write_counter_to_file(counter, file):
    with open(file, 'w+') as f:
        for element, count in counter.items():
            f.write("{}\t{}\n".format(element, count))

def read_counter_from_file(file):
    counter = Counter()
    with open(file) as f:
        for line in f:
            key, count = line.split('\t')
            counter[key] = int(count)
    return counter

class MLE:
    def __init__(self, q_counter, e_counter):
        self.q_counter = q_counter
        self.e_counter = e_counter
        self.generate_tags_counter_and_symbolizer()
        self.build_q_matrix_and_possible_states_set()
        # self.build_s_matrix()



    def generate_tags_counter_and_symbolizer(self):
        t_counter = Counter()
        vocab = set()
        for word_tag, count in self.e_counter.items():
            word_tag = DataTools.str_to_wordtag(word_tag)
            t_counter[word_tag.tag] += count
            vocab.add(word_tag.word)
        self.symbolizer = DataTools.get_symbolizer(vocab)
        self.t_counter = t_counter
        self.tags = list(set(t_counter.keys()) - set([DataTools.START_SYM]))

    def build_q_matrix_and_possible_states_set(self):
        n = len(self.tags)
        self.q_matrix = np.zeros((n,n,n))
        self.possible_states = list()
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    self.q_matrix[i, j, k] = self.getQ(self.tags[i], self.tags[j], self.tags[k])
                    if self.q_matrix[i, j, k] > 0:
                        self.possible_states.append((i, j, k))

    # def build_s_matrix(self):
    #     start = DataTools.START_SYM
    #     n = len(self.tags)
    #     self.s_matrix = np.zeros((n,))
    #     for tag, i in enumerate(self.tags):
    #         self.s_matrix[i] = self.getQ(start, start, tag)

    #return the probability something will start with tag of index i
    # def getS(self, index):
    #     return self.s_matrix[index]

    def getTags(self):
        return self.tags

    #get q by (i, j, k) = state
    def getQi(self, state):
        return self.q3_matrix[state]


    def getQ(self, t1, t2=None, t3=None):
        try:
            if t3:
                return self.q_counter[concat((t1, t2, t3))] / self.q_counter[concat((t1, t2))]
            elif t2:
                return self.q_counter[concat((t1, t2))] / self.t_counter[t1]
            else:  # case for general n n-gram:'
                if len(t1[:-1]) == 1:
                    return self.q_counter[concat(t1)] / self.t_counter[concat(t1[:-1])]
                return self.q_counter[concat(t1)] / self.q_counter[concat(t1[:-1])]
        except:
            return 0


    def getE(self, word, tag):
        word = self.symbolizer(word)
        return self.e_counter[concat((word, tag))] / self.t_counter[tag]

def main(argv):
    #args: ass1-tagger_train q.mle e.mle
    prog , input_file, qmle_file, emle_file = argv
    dp = DataProcessor(n_gram=3, add_start_symbols=True)
    data = DataLoader.load_tagged_data(input_file)
    data = dp.preprocess(data)
    write_counter_to_file(count_qmle(data), qmle_file)
    write_counter_to_file(count_emle(data), emle_file)


if __name__ == '__main__':
    from sys import argv
    main(argv)
