import pickle
import nltk
from nltk.chunk.util import tree2conlltags, conlltags2tree
from nltk.corpus import conll2000


def findpostagidx(tokens):

    with open('german_raw_token.pkl', 'rb') as f:
        tokens_raw = pickle.load(f)
    postag_idx = []

    for i in range(len(tokens)):
        found = False
        for j in range(len(tokens_raw)):
            if tokens[i] == tokens_raw[j]:
                postag_idx.append(j)
                found = True
                break
        if not found:
            print(i, tokens[i])
    print(len(postag_idx))
    print(postag_idx)
    return postag_idx

def findpostag():
    with open('german_val2_token.pkl', 'rb') as f:
        tokens = pickle.load(f)
    postag_idx = findpostagidx(tokens)
    print(postag_idx)
    german_raw = 'deu_uft_8_pos.raw'
    with open(german_raw, 'r') as f:
        raw = f.read()
        raw_lst = raw.split('-DOCSTART- -X- -X-\n')
        raw = '\n'.join(raw_lst)
        raw_lst = raw.split('\n\n')
    postags = []
    print(raw_lst[1])
    # for idx, sent in enumerate(raw_lst[0:2]):
    for idx in postag_idx:
        sent = raw_lst[idx]
        sent_lst = sent.split('\n')
        # print(sent_lst)
        parentheses = ['(', ')', '[', ']', '{', '}']
        if len(sent_lst) > 1:
            postag = []
            for words in sent_lst:
                words_lst = words.split(' ')
                if words_lst[2][0] == 'O':
                    continue
                if any(ele in words_lst[0] for ele in parentheses):
                    continue
                postag.append(' '.join(words_lst))
            postags.append('\n'.join(postag))

    return postags

def check():
    postags = findpostag()
    with open('german_val_postag.txt', 'w') as f:
        for i in postags:
            f.write(i)
            f.write('\n')
            f.write('\n')
    tokens_check= []
    for idx, sent in enumerate(postags):
        token = []
        sent_lst = sent.split('\n')
        # print(sent_lst)
        if len(sent_lst) > 1:
            for words in sent_lst:
                words_lst = words.split(' ')
                # print(words_lst)
                token.append(words_lst[0])
            tokens_check.append(token)
    with open('german_val2_token.pkl', 'rb') as f:
        tokens = pickle.load(f)
    for i in range(2000):
        if tokens[i] != tokens_check[i]:
            print(i)
            # print(tokens[i])
            # print(tokens_check[i])
    print(tokens == tokens_check)


def prepareForNLP(sentences):
    sentences = [' '.join(sent) for sent in sentences]
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

class BigramChunker(nltk.ChunkParserI):
    # chunker trained by conll2000
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in sent]
                      for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word,pos,BI) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        # print('chunktags', chunktags)
        conlltags = [(word, chunktag) for ((word,pos, BI),chunktag)
                     in zip(sentence, chunktags)]
        return conlltags


if __name__ == '__main__':
    train_sents = conll2000.chunked_sents('train.txt')
    german_train = 'german_train_postag.pkl'
    with open(german_train, 'rb') as f:
        german_train = pickle.load(f)
    # print(german_train)
    bigram_chunker = BigramChunker(german_train)
    german_test = 'german_test_postag.pkl'
    with open(german_test, 'rb') as f:
        german_test = pickle.load(f)

    with open('german_nltk_output.txt', "w") as f:
        for i, sent in enumerate(german_test):
            token_accuracy = 0
            if len(sent) <= 2:
                continue
            IOB_tag2 = bigram_chunker.parse(sent)
            hasNone = False
            for j in range(len(IOB_tag2)):
                if IOB_tag2[j][-1] is None:
                    hasNone = True
                    break
            if not hasNone:
                f.write("x y B B")
                f.write("\n")
                for j in range(len(IOB_tag2) - 1):
                    f.write("x y ")
                    f.write(IOB_tag2[j][-1][0])
                    f.write(' ')
                    f.write(sent[j][-1][0])
                    f.write("\n")
                f.write("\n")

    # check()

