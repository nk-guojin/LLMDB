import os.path as osp
import torch
import openai
import time
import numpy as np
from tqdm import tqdm
import pylcs
import html
import random
from utils import get_api_responses
import re
import json


class BaseModel:
    def __init__(self, user_his, candidates, item_names, users, ground_truth, dataset):
        self.user_his = user_his
        self.candidates = candidates
        self.item_names = item_names
        self.user_list = users
        self.batch_size = len(users)
        self.ground_truth = ground_truth
        self.dataset = dataset

    def get_item_names(self):
        user_his_name_list = []
        candidates_name_list = []
        truth_name = []
        for user_id in self.user_list:
            user_his_name = []
            candidates_name = []

            for his in self.user_his[user_id]:
                user_his_name.append(self.item_names[str(his)])
            for candidate in self.candidates[user_id]:
                candidates_name.append(self.item_names[str(candidate)])

            user_his_name_list.append(user_his_name)
            candidates_name_list.append(candidates_name)
        for i in range(self.batch_size):
            truth_name.append(self.item_names[str(self.ground_truth[i])])
        return user_his_name_list, candidates_name_list, truth_name

    def request_api(self, prompt_list):
        request_list = []
        for message in tqdm(prompt_list):
            request_list.append(get_api_responses(message))
            time.sleep(0.5)
        return request_list
    
    def parse_text(self, request_text):
        pattern = r'(?:\d+\.\s*)?([^\[\]\n]+)(?=\n|\[|$)'
        matches = re.findall(pattern, request_text, flags=re.DOTALL)

        predict_text = [m.strip() for m in matches]
        return predict_text
    
    def parse_predict(self, predict_text, truth):
        predict = [False for _ in range(len(predict_text))]
        for i in range(len(predict_text)):
            if (predict_text[i] in truth) or (truth in predict_text[i]) or (pylcs.lcs_sequence_length(truth, predict_text[i]) > 0.9 * len(predict_text[i])):
                predict[i] = True
                break
        return predict