# from utils.yk import get_actions, get_nonbinary_spans, get_tags_tokens_lowercase
import pickle


def combine_sent(path):
    file = open(path, "r")
    lines = file.readlines()
    # print(lines)
    all_sents = []
    i = 0
    while i < len(lines):
        print(i)
        line = lines[i]
        raw_tokens = []
        while line != '\n' and i+1 < len(lines):
            raw_tokens.append(line.split()[0])
            i += 1
            line = lines[i]
            # print(line)
        all_sents.append(raw_tokens)
        i += 1
    print(all_sents)
    return all_sents


def readfromAnup(token_path):
    with open(token_path, 'rb') as f:
        data = pickle.load(f)
        all_sents = []
        for i in data:
            all_sents.append(" ".join(i))
    return all_sents


class Dataset(object):
    def __init__(self, path, tokenizer=None):
        self.path = path
        self.tokenizer = tokenizer

        self.cnt = 0
        self.sents = []
        self.raw_tokens = []
        self.tokens = []
        self.masks = []
        self.gold_spans = []
        self.gold_tags = []
        self.gold_trees = []

        flatten = lambda l: [item for sublist in l for item in sublist]

        # with open(path, 'r') as f:
        #     lines = f.readlines()
        all_sents = readfromAnup(path)

        for sent in all_sents:
            # raw_tokens = get_tags_tokens_lowercase(line)[1]
            raw_tokens = sent.split()
            # actions = get_actions(line)
            self.cnt += 1
            self.sents.append(sent)
            self.raw_tokens.append(raw_tokens)
            self.tokens.append(self.tokenizer.tokenize(sent))
            mask = [len(self.tokenizer.tokenize(w)) * [i]
                    for i, w in enumerate(sent.split())]
            self.masks.append(flatten(mask))
