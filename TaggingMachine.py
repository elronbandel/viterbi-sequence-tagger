from old_code.data_tools import WordTag


class TaggingMachine:
    def __init__(self, tagger):
        self.tagger = tagger

    def tag(self, text):
        return [[WordTag(word, tag) for word, tag in zip(line, self.tagger.tag(line)) ] for line in text]

