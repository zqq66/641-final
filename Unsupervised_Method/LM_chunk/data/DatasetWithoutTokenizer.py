from utils.yk import get_actions, get_nonbinary_spans, get_tags_tokens_lowercase, clean_number
import re


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

        with open(path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            # print(line)
            tags, sent, sent_lower = get_tags_tokens_lowercase(line)
            # print(sent_lower)
            # sentences only
            actions = get_actions(line)
            self.cnt += 1
            self.raw_tokens.append(sent_lower)
            # print(self.raw_tokens)
            # print(self.sents)
            # self.tokens.append(self.tokenizer.tokenize(sent))
            # mask = [len(self.tokenizer.tokenize(w)) * [i]
            #         for i, w in enumerate(sent.split())]
            # self.masks.append(flatten(mask))
            gold_spans, gold_tags, _, _ = get_nonbinary_spans(actions)
            sent = [clean_number(w) for w in sent_lower]
            # print(sent)
            self.sents.append(" ".join(sent))
            self.gold_spans.append(gold_spans)
            self.gold_tags.append(gold_tags)
            self.gold_trees.append(line.strip())
        # print(self.sents)
