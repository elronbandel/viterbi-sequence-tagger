import numpy as np
import operator

from DataTools import DataProcessor, DataLoader
import DataTools
from collections import namedtuple

n_gram = 3

def find_rare_words(data, rarity_precentage=0.3):
    words_counter = DataTools.count_words(data)
    threshold = int(rarity_precentage * len(words_counter))
    rares = map(operator.itemgetter(0), words_counter.most_common()[-threshold:])
    return set(rares)


def add_prefix_suffix(dictionary, word):
    for i in range(min(4, len(word))):
        dictionary["pre{}".format(i+1)] = word[:i+1]
        dictionary["suf{}".format(i+1)] = word[-i-1:]


def extract_features_from_line(line, rares_set):
    line_features = []
    offset = n_gram - 1
    for i in range(offset, len(line) - offset):
        word, tag = line[i]
        if word in rares_set:
            word_features = {
                "W_i-2" : line[i - 2].word,
                "W_i-1" : line[i - 1].word,
                "W_i+1" : line[i + 1].word,
                "W_i+2" : line[i + 2].word,
                "T_i-1" : line[i - 1].tag,
                "T_i-1,i-2" : ",".join([line[i - 2].tag, line[i - 1].tag]),
                "chyp" : '-' in word,
                "cnum" : any(char.isdigit() for char in word),
                "cupp" : any(char.isupper() for char in word)
            }
            add_prefix_suffix(word_features, word)
        else:
            word_features = {
                "W_i-2" : line[i].word,
                "W_i-1" : line[i-1].word,
                "W_i" : line[i].word,
                "W_i+1": line[i+1].word,
                "W_i+2" : line[i+2].word,
                "T_i-1" : line[i-1].tag,
                "T_i-1,i-2": ",".join([line[i-2].tag, line[i-1].tag]),
            }
        line_features.append((tag, word_features))
    return line_features


def extract_features_from_data(data):
    rares = find_rare_words(data)
    features = []
    for line in data:
        features += extract_features_from_line(line, rares)
    return features

def write_features_to_file(features, file):
    with open(file, 'w+') as f:
        for tag, feat in features:
            f.write(tag + " " + " ".join("{}:{}".format(key, val) for key,val in feat.items()) + "\n")



def main(argv):
    #args: ass1-tagger-train features_file
    prog , input_file, output_file = argv
    dp = DataProcessor(n_gram=3, add_start_symbols=True, add_end_symbols=True, manipulate_unknowns=False)
    data = DataLoader.load_tagged_data(input_file)
    data = dp.preprocess(data)
    features = extract_features_from_data(data)
    write_features_to_file(features, output_file)

if __name__ == '__main__':
    from sys import argv
    main(argv)




