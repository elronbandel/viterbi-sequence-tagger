import re
from collections import namedtuple, Counter

UNK_SYM = "*UNK*"
NUM_SYM = "*NUM*"
START_SYM = "*START*"
SymbolsRegex = {
    "*NUM*" : re.compile('^(([1-9]+[0-9]*)|([0])|([1-9]([0-9]?)([0-9]?)(([,][0-9]{3})*)))([.][0-9]*)?$'),
    "*CAP*" : re.compile('^[A-Z]'),
    "*ING*" : re.compile(".*ing$"),
    "*HYP*" : re.compile('.*[-].*'),
    "*CNUM*" : re.compile('.*[0-9].*'),
    "*UNK2*" : re.compile('^.{1,2}$'),
    "*UNK4*" : re.compile('^.{3,4}$'),
    "*UNK8*" : re.compile('^.{5,6,7,8}$')
}

WordTag = namedtuple('WordTag', ['word', 'tag'])


class DataProcessor:

    #all the data tools correspond to the 2 initial parameters
    def __init__(self, n_gram=3, add_start_symbols=False):
        self.n_gram = n_gram
        self.start_sym = add_start_symbols


    def preprocess(self, data):
        data = apply_symbolizer(get_symbolizer(), data)
        if self.start_sym:
            data = add_start_symbol_to_data(data, n_gram=self.n_gram)
        vocab = build_vocab(data)
        return maniplulate_data_to_fit_vocab(data, vocab)

    def preprocess_untagged(self, data):
        if self.start_sym:
            data = add_start_symbol_to_untagged_data(data, self.n_gram)
        return data


    def deprocess(self, data):
        if self.start_sym:
            data = delete_start_symbol_from_data(data, self.n_gram)
        return data


class DataLoader:

    @staticmethod
    def load_tagged_data(file):
        # in case there is more than 2 cells it means there was more than one '/' in the original token
        word_tag = lambda token: WordTag('/'.join(token[:-1]), token[-1])
        with open(file, 'r') as f:
            return [[word_tag(token.split('/')) for token in line.split()] for line in f]

    @staticmethod
    def save_tagged_data(data, file):
        with open(file, 'w+') as f:
            for line in data:
                f.write(' '.join(['/'.join([word, tag]) for word, tag in line]) + '\n')

    @staticmethod
    def load_untagged_data(file, n_gram=3):
        with open(file, 'r') as f:
            return [line.split() for line in f]







def str_to_wordtag(str):
    word, tag = str.split()
    return WordTag(word, tag)

def words(sentence):
    return [token.word for token in sentence]


def tags(sentence):
    return [token.tag for token in sentence]


def count_words(data):
    counter = Counter()
    for sentence in data:
        for word in words(sentence):
            counter[word] += 1
    return counter

def count_tags(data):
    counter = Counter()
    for sentence in data:
        for tag in tags(sentence):
            counter[tag] += 1
    return counter


#####################################################################################
## pre-processing
#####################################################################################


def is_num(string):
    # every number of the form: X,XXX,XXX.XXXXX
    regex = re.compile('^(([1-9]+[0-9]*)|([0])|([1-9]([0-9]?)([0-9]?)(([,][0-9]{3})*)))([.][0-9]*)?$')
    return bool(regex.match(string))


def get_symbolizer(vocab=None):
    def symbolizer(word):
        if is_num(word):
            return NUM_SYM
        if vocab:
            if word not in vocab:
                for symbol, regex in SymbolsRegex.items():
                    if bool(regex.match(word)):
                        return symbol
                return UNK_SYM
        return word
    return symbolizer


def apply_symbolizer(symbolizer, data):
    def generate(word_tag):
        return WordTag(symbolizer(word_tag.word), word_tag.tag)
    return [[generate(word_tag) for word_tag in line] for line in data]


def add_start_symbol_to_line_of_word_tags(line, n_gram):
    t = n_gram - 1
    return ([WordTag(START_SYM, START_SYM)] * t) + line


def add_start_symbol_to_list_of_words(line, n_gram):
    t = n_gram - 1
    return ([START_SYM] * t) + line


def add_start_symbol_to_untagged_data(data, n_gram):
    return [add_start_symbol_to_list_of_words(line, n_gram) for line in data]


def add_start_symbol_to_data(data, n_gram):
    return [add_start_symbol_to_line_of_word_tags(line, n_gram) for line in data]

def delete_start_symbol_from_data(data, n_gram):
    t = n_gram - 1
    return [line[t:] for line in data]


def build_vocab(data, p=0.7):
    count = count_words(data)
    threshold = int(p * len(count))
    most_common = count.most_common()[:threshold]
    vocab = set(map(lambda pair: pair[0], most_common))
    return vocab

def maniplulate_data_to_fit_vocab(data, vocab):
    symbolizer = get_symbolizer(vocab)
    def filter(word_tag):
        return WordTag(symbolizer(word_tag.word), word_tag.tag)
    return [[filter(word_tag) for word_tag in line] for line in data]

