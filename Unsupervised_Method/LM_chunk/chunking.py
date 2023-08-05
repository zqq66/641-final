import argparse
import datetime
import logging
import os
import pickle
import heapq
from tqdm import tqdm

import torch
from transformers import *
from queue import PriorityQueue

from data.CoNLL_dataset import Dataset
from utils.measure import Measure
from utils.parser import not_coo_parser, parser
from utils.tools import set_seed, select_indices, group_indices
from utils.yk import get_actions, get_nonbinary_spans


class DualPriorityQueue(PriorityQueue):
    def __init__(self, maxPQ=False):
        PriorityQueue.__init__(self)
        self.reverse = -1 if maxPQ else 1

    def put(self, priority, data):
        PriorityQueue.put(self, (self.reverse * priority, data))

    def get(self, *args, **kwargs):
        priority, data = PriorityQueue.get(self, *args, **kwargs)
        return self.reverse * priority, data


MODELS = [#(BertModel, BertTokenizer, BertConfig, 'bert-base-cased'),
          (BertModel, BertTokenizer, BertConfig, 'bert-large-cased')]
          #(GPT2Model, GPT2Tokenizer, GPT2Config, 'gpt2')]
          # (GPT2Model, GPT2Tokenizer, GPT2Config, 'gpt2-medium'),
          # (RobertaModel, RobertaTokenizer, RobertaConfig, 'roberta-base'),
          # (RobertaModel, RobertaTokenizer, RobertaConfig, 'roberta-large'),
          # (XLNetModel, XLNetTokenizer, XLNetConfig, 'xlnet-base-cased'),
          # (XLNetModel, XLNetTokenizer, XLNetConfig, 'xlnet-large-cased')]


def evaluate(args):
    sent_len = dict()
    all_words_dist = dict()
    for model_class, tokenizer_class, model_config, pretrained_weights in MODELS:
        tokenizer = tokenizer_class.from_pretrained(
            pretrained_weights, cache_dir=args.lm_cache_path, force_download=True)

        if args.from_scratch:
            config = model_config.from_pretrained(pretrained_weights, force_download=True)
            config.output_hidden_states = True
            config.output_attentions = True
            model = model_class(config).to(args.device)
        else:
            model = model_class.from_pretrained(
                pretrained_weights,
                cache_dir=args.lm_cache_path,
                output_hidden_states=True,
                output_attentions=True).to(args.device)

        with torch.no_grad():
            test_sent = tokenizer.encode('test', add_special_tokens=False)
            token_ids = torch.tensor([test_sent]).to(args.device)
            all_hidden, all_att = model(token_ids)[-2:]
            n_layers = len(all_att)
            n_att = all_att[0].size(1)
            n_hidden = all_hidden[0].size(-1)

        measure = Measure(n_layers, n_att)
        data = Dataset(path=args.data_path, tokenizer=tokenizer)

        for idx, s in enumerate(data.sents):
            print("sentences",idx, s)
            raw_tokens = data.raw_tokens[idx]
            # print("raw_tokens", raw_tokens)
            tokens = data.tokens[idx]
            # print("tokens", tokens)
            if len(raw_tokens) < 2:
                data.cnt -= 1
                continue
            token_ids = tokenizer.encode(s, add_special_tokens=False)
            token_ids_tensor = torch.tensor([token_ids]).to(args.device)
            with torch.no_grad():
                all_hidden, all_att = model(token_ids_tensor)[-2:]
            all_hidden, all_att = list(all_hidden[1:]), list(all_att)

            # hidden layer, attention distribution from LM
            # (n_layers, seq_len, hidden_dim)
            all_hidden = torch.cat([all_hidden[n] for n in range(n_layers)], dim=0)
            # (n_layers, n_att, seq_len, seq_len)
            all_att = torch.cat([all_att[n] for n in range(n_layers)], dim=0)

            if len(tokens) > len(raw_tokens):
                th = args.token_heuristic
                if th == 'first' or th == 'last':
                    mask = select_indices(tokens, raw_tokens, pretrained_weights, th)
                    assert len(mask) == len(raw_tokens)
                    all_hidden = all_hidden[:, mask]
                    all_att = all_att[:, :, mask, :]
                    all_att = all_att[:, :, :, mask]
                else:
                    # mask = torch.tensor(data.masks[idx])
                    try:
                        mask = group_indices(tokens, raw_tokens, pretrained_weights)
                    except:
                        print(idx, s)
                        continue
                    raw_seq_len = len(raw_tokens)
                    all_hidden = torch.stack(
                        [all_hidden[:, mask == i].mean(dim=1)
                         for i in range(raw_seq_len)], dim=1)
                    all_att = torch.stack(
                        [all_att[:, :, :, mask == i].sum(dim=3)
                         for i in range(raw_seq_len)], dim=3)
                    all_att = torch.stack(
                        [all_att[:, :, mask == i].mean(dim=2)
                         for i in range(raw_seq_len)], dim=2)

            l_hidden, r_hidden = all_hidden[:, :-1], all_hidden[:, 1:]
            l_att, r_att = all_att[:, :, :-1], all_att[:, :, 1:]
            # find syntactic distance
            syn_dists = measure.derive_dists(l_hidden, r_hidden, l_att, r_att)
            # for i in syn_dists.keys():
            #     if i in all_words_dist.keys():
            #         all_words_dist[i].append([j.item() for j in torch.mean(syn_dists[i], dim=0)])
            #     else:
            #         all_words_dist[i] = [[j.item() for j in torch.mean(syn_dists[i], dim=0)]]
            # print("syn_dists", syn_dists["avg_hellinger"])
            for i in range(1, syn_dists["avg_hellinger"].shape[0] + 1):
                if i in all_words_dist.keys():
                    all_words_dist[i].append([j.item() for j in syn_dists["avg_hellinger"][i-1, :]])
                else:
                    all_words_dist[i] = [[j.item() for j in syn_dists["avg_hellinger"][i-1, :]]]
            # print(all_words_dist)
            sent_len[idx] = syn_dists["avg_hellinger"].shape[1]
            # findcuts(args, all_words_dist, sent_len)
    with open('all_words_dist.pkl', 'wb') as f:
        pickle.dump(all_words_dist, f)
    with open('sent_len.pkl', 'wb') as f:
        pickle.dump(sent_len, f)
    return all_words_dist, sent_len


def findcuts(args, all_words_dist, sent_len):

    tag_data = args.datatag_path
    with open(tag_data, 'rb') as f:
        data = pickle.load(f)
        num_sent = len(data)
        print(num_sent)
        numcuts = 0
        for i in data:
            numcuts += i.count("B") - 1

    # print("all_words_dist", all_words_dist.keys())
    # print("sent_len", sent_len)
    for func in all_words_dist.keys():
        cuts_dic = []
        maxQ = DualPriorityQueue(maxPQ=True)
        for sent in range(len(sent_len.keys())):
            lst = all_words_dist[func][sent]
            # print(lst[0])
            # push distances into priority queue
            # print(len(lst))
            for i in range(len(lst)):
                maxQ.put(lst[i], (sent, i))
        # find top k distances
        for j in range(numcuts):
            value, (sent, phrase) = maxQ.get()
            cuts_dic.append((sent, phrase))
            # print("value, idx", value, (sent, phrase))
        print('cuts_dic', cuts_dic)
        print('sent_len', sent_len)

        with open(f'{args.result_path}/output_review_{func}.txt', "w") as f:
            for i in range(num_sent):
                f.write("x y B B")
                f.write("\n")
                for j in range(sent_len[i]):
                    if (i, j) in cuts_dic:
                        f.write("x y B ")
                    else:
                        f.write("x y I ")
                    print(i, j, len(data[i]))
                    f.write(data[i][j+1])
                    f.write("\n")
                f.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',
                        default="/Users/qianqiu/Documents/CMPUT499/UC/data/review/review_test_token.pkl", type=str) # twitter/twitter_test_token.pkl
    parser.add_argument('--datatag-path',
                        default="/Users/qianqiu/Documents/CMPUT499/UC/data/review/review_test_tag.pkl",
                        type=str)
    parser.add_argument('--result-path', default='/Users/qianqiu/Documents/CMPUT499/Unsupervised_Parsing/trees_from_transformers/chunking_output/LM', type=str)
    parser.add_argument('--lm-cache-path',
                        default='/Users/qianqiu/Documents/CMPUT499/Unsupervised_Parsing/trees_from_transformers/data/transformers', type=str)
    parser.add_argument('--from-scratch', default=False, action='store_true')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--bias', default=0.0, type=float,
                        help='the right-branching bias hyperparameter lambda')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--token-heuristic', default='mean', type=str,
                        help='Available options: mean, first, last')
    parser.add_argument('--use-not-coo-parser', default=False,
                        action='store_true',
                        help='Turning on this option will allow you to exploit '
                             'the NOT-COO parser (named by Dyer et al. 2019), '
                             'which has been broadly adopted by recent methods '
                             'for unsupervised parsing. As this parser utilizes'
                             ' the right-branching bias in its inner workings, '
                             'it may give rise to some unexpected gains or '
                             'latent issues for the resulting trees. For more '
                             'details, see https://arxiv.org/abs/1909.09428.')

    args = parser.parse_args()

    setattr(args, 'device', f'cuda:{args.gpu}'
    if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    dataset_name = args.data_path.split('/')[-1].split('.')[0]
    parser = '-w-not-coo-parser' if args.use_not_coo_parser else ''
    pretrained = 'scratch' if args.from_scratch else 'pretrained'
    result_path = f'{args.result_path}'  # {dataset_name}-{args.token_heuristic}'
    # result_path += f'-{pretrained}-{args.bias}{parser}'
    setattr(args, 'result_path', result_path)
    set_seed(args.seed)
    logging.disable(logging.WARNING)
    print('[List of arguments]')
    for a in args.__dict__:
        print(f'{a}: {args.__dict__[a]}')

    all_words_dist, sent_len = evaluate(args)
    findcuts(args, all_words_dist, sent_len)

    # scores = evaluate(args)
    # with open(f'{args.result_path}/scores.pickle', 'wb') as f:
    #     pickle.dump(scores, f)


if __name__ == '__main__':
    main()
