import pickle


def preprocess(input, output):
    with open(input, 'r') as f:
        test = f.readlines()
        idx = 0
        length = len(test)
        sents = []
        while idx < length:
            sent = []
            while test[idx] != '\n' and idx < length:
                lst = test[idx].split()
                if lst[-1] != 'O':
                    sent.append(tuple(test[idx].split()))

                idx += 1
            sent.append(tuple(test[idx-1].split()))
            idx += 1
            sents.append(sent)
        print(sents[693])
    with open(output, 'wb') as f:
        pickle.dump(sents, f)


if __name__ == '__main__':
    test_data = 'review_test_postag.txt'
    train_data = 'review_train_postag.txt'
    preprocess(test_data, 'review_test_postag.pkl')
    print(test_data)
    preprocess(train_data, 'review_train_postag.pkl')