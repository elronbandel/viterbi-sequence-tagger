from collections import Counter
import data_tools


def count_qmle(data):
    counter = Counter()
    for line in data:
        for word_tag in line:
            counter[' '.join(word_tag)] += 1
    return counter

def count_emle(data):
    counter = Counter()
    for line in data:
        tags = data_tools.tags(line)
        for pair in zip(tags, tags[1:]):
            counter[' '.join(pair)] += 1
        for triplet in zip(tags, tags[1:], tags[2:]):
            counter[' '.join(triplet)] +=1
    return counter


def write_counter_to_file(counter, file):
    with open(file, 'w+') as f:
        for element, count in counter.items():
            f.write("{}\t{}\n".format(element, count))


def main(argv):
    prog , input_file, qmle_file, emle_file = argv
    data = data_tools.load(input_file)
    data = data_tools.preprocess(data)
    write_counter_to_file(count_qmle(data), qmle_file)
    write_counter_to_file(count_emle(data), emle_file)


if __name__ == '__main__':
    from sys import argv
    main(argv)
