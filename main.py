import data_ml100k as ml100k
import data_lastfm as lastfm
import data_steam as steam
import data_games as games
import os
from models.Direct import DirectModel
from models.NIR import NirModel
from models.Iteration import IterationModel
from models.Prefer import PreferModel
from models.LLMDB import LLMDB
import argparse

import evaluate
import random
import popularity

def get_truth_text(item_names, ground_truth):
    truth_name = []
    for i in range(len(ground_truth)):
        truth_name.append(item_names[str(ground_truth[i])])
    return truth_name

def main(model_name, dataset_name, user_num, **kwargs):
    random.seed(2025)
    DATASET_NAME = dataset_name
    MODEL_NAME = model_name
    USER_NUM = user_num

    print(DATASET_NAME)
    print(MODEL_NAME)

    if (DATASET_NAME == "lastfm"):
        directory = 'data/lastfm'
        file_path = os.path.join(directory, DATASET_NAME)
        lastfm.sort_data(file_path)
        train, valid, test = lastfm.get_data(file_path)
        lastfm.get_candidate(file_path, num_user=USER_NUM)
        candidates, users = lastfm.generate_candidate(file_path, test_data=test, num_candidate=20)
        item_names = lastfm.load_text(file_path)
        item_pop = lastfm.item_popularity(file_path, item_names)
        user_his, ground_truth = lastfm.inter_his(file_path, history_num=20)
    elif (DATASET_NAME == "ml-100k"):
        directory = 'data/ml-100k'
        file_path = os.path.join(directory, DATASET_NAME)
        ml100k.sort_data(file_path)
        train, valid, test = ml100k.get_data(file_path)
        ml100k.get_candidate(file_path, num_user=USER_NUM)
        candidates, users = ml100k.generate_candidate(file_path, test_data=test, num_candidate=20)
        item_names = ml100k.load_text(file_path)
        item_pop = ml100k.item_popularity(file_path, item_names)
        user_his, ground_truth = ml100k.inter_his(file_path, history_num=20)
    elif (DATASET_NAME == "ml-1m"):
        directory = 'data/ml-1m'
        file_path = os.path.join(directory, DATASET_NAME)
        ml100k.sort_data(file_path)
        train, valid, test = ml100k.get_data(file_path)
        ml100k.get_candidate(file_path, num_user=USER_NUM)
        candidates, users = ml100k.generate_candidate(file_path, test_data=test, num_candidate=20)
        item_names = ml100k.load_text(file_path)
        item_pop = ml100k.item_popularity(file_path, item_names)
        user_his, ground_truth = ml100k.inter_his(file_path, history_num=20)
    elif (DATASET_NAME == "steam"):
        directory = 'data/steam'
        file_path = os.path.join(directory, DATASET_NAME)
        steam.sort_data(file_path)
        train, valid, test = steam.get_data(file_path)
        steam.get_candidate(file_path, num_user=USER_NUM)
        candidates, users = steam.generate_candidate(file_path, test_data=test, num_candidate=20)
        item_names = steam.load_text(file_path)
        item_pop = steam.item_popularity(file_path, item_names)
        user_his, ground_truth = steam.inter_his(file_path, history_num=20)
    elif (DATASET_NAME == "Amazon-Games"):
        directory = 'data/Amazon-Games'
        file_path = os.path.join(directory, DATASET_NAME)
        games.process_amazon_data(file_path)
        games.sort_data(file_path)
        train, valid, test = games.get_data(file_path)
        games.get_candidate(file_path, num_user=USER_NUM)
        candidates, users = games.generate_candidate(file_path, test_data=test, num_candidate=20)
        item_names = games.load_text(file_path)
        item_pop = games.item_popularity(file_path, item_names)
        user_his, ground_truth = games.inter_his(file_path, history_num=20)

    # 进行预测
    if (MODEL_NAME == "direct"):
        model = DirectModel(user_his, candidates, item_names, users, ground_truth, DATASET_NAME)
        predict_list, predict_text_list = model.predict_rank()
    elif (MODEL_NAME == "nir"):
        model = NirModel(user_his, candidates, item_names, users, ground_truth, DATASET_NAME)
        predict_list, predict_text_list = model.predict_rank()
    elif (MODEL_NAME == "prefer"):
        model = PreferModel(user_his, candidates, item_names, users, ground_truth, DATASET_NAME)
        predict_list, predict_text_list = model.predict_rank()
    elif (MODEL_NAME == "iter"):
        model = IterationModel(user_his, candidates, item_names, users, ground_truth, DATASET_NAME)
        predict_list, predict_text_list = model.predict_rank()
    elif (MODEL_NAME == "LLMDB"):
        model = LLMDB(user_his, candidates, item_names, users, ground_truth, DATASET_NAME)
        predict_list, predict_text_list = model.predict_rank()

    # 进行评估
    recall, ndcg = evaluate.evaluate(predict_list)
    print("recall: ", [round(i, 3) for i in recall])
    print("ndcg: ", [round(i, 3) for i in ndcg])

    # 计算流行度
    pop = popularity.popularity(predict_text_list, item_pop)
    print("pop: ", [round(i, 3) for i in pop])

    gt_pop = popularity.get_truth_pop(get_truth_text(item_names, ground_truth), item_pop)
    print("gt_pop: ", gt_pop)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, default="LLMDB")
    parser.add_argument('-d', type=str, default='ml-100k')
    parser.add_argument('-u', type=int, default='200')
    args, unparsed = parser.parse_known_args()
    main(args.m, args.d, args.u)