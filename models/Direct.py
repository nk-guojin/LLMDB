from .base import BaseModel
import os.path as osp
import torch
import openai
import time
import numpy as np
from tqdm import tqdm
import pylcs
import html
import random
import re
import json

class DirectModel(BaseModel):
    def __init__(self, user_his, candidates, item_names, users, ground_truth, dataset):
        super().__init__(user_his, candidates, item_names, users, ground_truth, dataset)

    def build_prompt(self, user_his_name, candidates_name):
        if self.dataset == "lastfm":
            prompt = f"You are a music recommender system.\n" \
                    f"I've listened musics from the following artists in the past in order:\n{user_his_name}\n\n" \
                    f"Note that my most recently listened artist is {user_his_name[-1]}.\n" \
                    f"Now there are {len(candidates_name)} candidate artists that I can lsiten to next:\n{candidates_name}\n" \
                    f"Please rank these {len(candidates_name)} artists by measuring the possibilities that I would like to listen next most, according to my listen history. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. Rank directly, you can only generate movie names. You MUST rank the given candidate artists. You can not generate artists that are not in the given candidate list."
        elif self.dataset == "ml-100k" or self.dataset == "ml-1m":
            prompt = f"You are a movie recommender system on movielens dataset.\n" \
                    f"I've watched the following movies in the past in order:\n{user_his_name}\n\n" \
                    f"Note that my most recently watched movie is {user_his_name[-1]}.\n" \
                    f"Now there are {len(candidates_name)} candidate movies that I can watch next:\n{candidates_name}\n" \
                    f"Please rank these {len(candidates_name)} movies by measuring the possibilities that I would like to watch next most, according to my watching history. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. Rank directly, you can only generate movie names. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."
        elif self.dataset == "steam" or self.dataset == "Amazon-Games":
            prompt = f"You are a game recommender system.\n" \
                    f"I've played the following games in the past in order:\n{user_his_name}\n\n" \
                    f"Note that my most recently played game is {user_his_name[-1]}.\n" \
                    f"Now there are {len(candidates_name)} candidate games that I can play next:\n{candidates_name}\n" \
                    f"Please rank these {len(candidates_name)} games by measuring the possibilities that I would like to play next most, according to my playing history. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. Rank directly, you can only generate game names. You MUST rank the given candidate games. You can not generate games that are not in the given candidate list."

        return prompt

    def predict_rank(self):
        prompt_list = []
        user_his_name_list, candidates_name_list, truth_name = self.get_item_names()
        for i in range(self.batch_size):
            user_his_name = user_his_name_list[i]
            candidates_name = candidates_name_list[i]
            prompt = self.build_prompt(user_his_name, candidates_name)
            prompt_list.append(prompt)
        request_list = self.request_api(prompt_list)

        results_data = []
        for i in range(self.batch_size):
            result_json = {"PID": i,
                    "ground truth": truth_name[i],
                    "Input": prompt_list[i],
                    "Predictions": request_list[i]}
            results_data.append(result_json)

        file_dir = f"./logs/{self.dataset}/Direct_{self.dataset}.json"
        with open(file_dir, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        predict_text_list = []
        for request_text in request_list:
            predict_text = self.parse_text(request_text)
            predict_text_list.append(predict_text)

        predict_list = []
        for i in range(self.batch_size):
            predict = self.parse_predict(predict_text_list[i], truth_name[i])
            predict_list.append(predict)
        return predict_list, predict_text_list
