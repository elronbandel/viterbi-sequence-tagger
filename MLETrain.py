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


def main(argv):
    #args: data/ass1-tagger-train q.mle e.mle
    prog , input_file, qmle_file, emle_file = argv
    dp = DataProcessor(n_gram=3, add_start_symbols=True)
    data = DataLoader.load_tagged_data(input_file)
    data = dp.preprocess(data)
    write_counter_to_file(count_qmle(data), qmle_file)
    write_counter_to_file(count_emle(data), emle_file)


if __name__ == '__main__':
    from sys import argv
    main(argv)
