import DataTools
from DataTools import DataLoader




def main():
    dev = DataLoader.load_tagged_data("data/ass1-tagger-dev")
    pred = DataLoader.load_tagged_data("ass1-tagger-dev-output")
    correct = 0
    sum = 0
    for line_dev, line_pred in zip(dev, pred):
        for word_tag_dev, word_tag_pred in zip(line_dev, line_pred):
            sum += 1
            if word_tag_dev.tag == word_tag_pred.tag:
                correct += 1
    print("accuracy: {}".format(correct/sum))

if __name__ == '__main__':
    main()