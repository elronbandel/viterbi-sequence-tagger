from MLETrain import MLE, read_counter_from_file
import data_tools
from TaggingMachine import TaggingMachine

class GreedyTag:
    def __init__(self, mle):
        self.mle = mle

    def tag(self, line):
        raise NotImplemented


def main(argv):
    prog , input_file, qmle_file, emle_file, output_file, extra_file = argv
    q_counter = read_counter_from_file(qmle_file)
    e_counter = read_counter_from_file(emle_file)
    mle = MLE(q_counter, e_counter)
    gt = GreedyTag(mle)
    tagger = TaggingMachine(gt)
    input = data_tools.load_untagged_data(input_file)
    output = tagger.tag(input)
    data_tools.save_tagged_data(output, output_file)




if __name__ == '__main__':
    from sys import argv
    main(argv)