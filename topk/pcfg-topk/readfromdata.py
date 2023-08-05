import pickle

topk_file = './topk/top5_span_trees.pkl'
chunk_file = './topk/ptb_pred_chunk_span.pkl'
output_file = './topk/top5_best_span.pkl'
max_file = './topk/max_span.pkl'
argmax_file = './top5_argmax.pkl'

# with open(chunk_file, 'rb') as f:
#     chunk_span_sets = pickle.load(f)
#     print(chunk_span_sets[5])
#
# with open(topk_file, 'rb') as f:
#     pred_span_sets = pickle.load(f)
#     best_spans = []
#     max_spans = []
#     # print(pred_span_sets[1][0])
#     for idx in range(len(pred_span_sets)):
#         max_spans.append(pred_span_sets[idx][0])
#         chunk_sent = chunk_span_sets[idx]
#         best_span = set()
#         max_match = -1
#         sent = pred_span_sets[idx]
#         for spans in pred_span_sets[idx]:
#             set_spans = set(spans)
#             set_chunk = set(chunk_sent)
#             lst = set_spans.intersection(set_chunk)
#             # print('len(lst)', len(lst))
#             # print(set_chunk)
#             # print(spans)
#             if len(lst) == len(set_chunk):
#                 best_span = spans
#                 break
#             if len(lst) > max_match:
#                 max_match = len(lst)
#                 best_span = spans
#         # print('best_span', best_span)
#         best_spans.append(best_span)
#     print(best_spans[5])
#
#
# with open(output_file, 'wb') as f:
#     pickle.dump(best_spans, f)
#
# with open(max_file, 'wb') as f:
#     pickle.dump(max_spans, f)

with open(argmax_file, 'rb') as f:
    argmax = pickle.load(f)
    print(len(argmax[3]))




