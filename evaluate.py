import numpy as np

TOPK = [1, 5, 10, 20]

def recall_k(predict_k, predict_len):
    hit = 0
    for predict in predict_k:
        if True in predict:
            hit = hit + 1
    recall = hit /predict_len    
    return recall

def idcg_k(k):
    idcg = 1 / np.log2(2)
    return idcg

def dcg_k(predict_k, predict_len):
    dcg = 0
    for predict in predict_k:
        for i in range(1, len(predict) + 1):
            if predict[i - 1]:
                dcg += 1 / np.log2(i + 1)
    return dcg

def ndcg_k(predict_k, predict_len, k):
    dcg = dcg_k(predict_k, predict_len)
    idcg = idcg_k(k)
    ndcg = dcg / (idcg * predict_len)
    return ndcg

def evaluate(predict_list):
    recall_list = []
    ndcg_list = []
    for k in TOPK:
        predict_k = [[predict[i] for i in range(min(k, len(predict)))] for predict in predict_list]
        predict_len = len(predict_k)
        recall_list.append(recall_k(predict_k, predict_len))
        ndcg_list.append(ndcg_k(predict_k, predict_len, k))
    return recall_list, ndcg_list
