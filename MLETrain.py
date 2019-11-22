from collections import Counter
import data_tools


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
        tags = data_tools.tags(line)
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
        self.generate_tags_counter()

    def generate_tags_counter(self):
        t_counter = Counter()
        for word_tag, count in self.e_counter.elements():
            t_counter[tag(word_tag)] += count
        self.t_counter = t_counter

    def getQ(self, t1, t2, t3=None):
        if t3:
            return self.q_counter[concat((t1, t2, t3))] / self.q_counter[concat((t1, t2))]
        else:
            return self.q_counter[concat((t1, t2))] / self.q_counter[t1]

    def getE(self, word, tag):
        return self.e_counter[concat((word, tag))] / self.q_counter[tag]

def main(argv):
    prog , input_file, qmle_file, emle_file = argv
    data = data_tools.load_tagged_data(input_file)
    data = data_tools.preprocess(data)
    write_counter_to_file(count_qmle(data), qmle_file)
    write_counter_to_file(count_emle(data), emle_file)


if __name__ == '__main__':
    from sys import argv
    main(argv)
