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

class PreferModel(BaseModel):
    def __init__(self, user_his, candidates, item_names, users, ground_truth, dataset):
        super().__init__(user_his, candidates, item_names, users, ground_truth, dataset)

    def build_prompt_his(self, user_his_name, candidates_name):
        if (self.dataset == "lastfm"):
            prompt_nir = f"You are a music recommender system.\n" \
                    f"I have listened to music from the following artists in the past in order:\n{user_his_name}\n\n" \
                    f"Note that my most recently listened artist is {user_his_name[-1]}.\n" \
                    f"Your task is to analyze a list of artests I have interacted with and describe the profile and the preferences of I in less than 300 words. Try to not be too broad (e.g. mention too many general categories). Do not mention specific artist names."
            
        elif (self.dataset == "ml-100k" or self.dataset == "ml-1m"):
            prompt_nir = f"You are a movie recommender system.\n" \
                    f"I have watched the following movies in the past in order:\n{user_his_name}\n\n" \
                    f"Note that my most recently watched movie is {user_his_name[-1]}.\n" \
                    f"Your task is to analyze a list of movies I have interacted with and describe the profile and the preferences of I in less than 300 words. Try to not be too broad (e.g. mention too many general categories such as action or comedy). Do not mention specific movie titles."
        elif (self.dataset == "steam" or self.dataset == "Amazon-Games"):
            prompt_nir = f"You are a game recommender system.\n" \
                    f"I have played the following games in the past in order:\n{user_his_name}\n\n" \
                    f"Note that my most recently played game is {user_his_name[-1]}.\n" \
                    f"Your task is to analyze a list of games I have interacted with and describe the profile and the preferences of I in less than 300 words. Try to not be too broad (e.g. mention too many general categories). Do not mention specific game titles."
        return prompt_nir

    def build_prompt_rec(self, user_his_name, candidates_name, user_his_answer):
        if (self.dataset == "lastfm"):
            prompt_nir = f"You are a music recommender system.\n" \
                    f"{user_his_answer}\n" \
                    f"Now there are {len(candidates_name)} candidate artists that I can listen to next:\n{candidates_name}\n" \
                    f"Please rank these {len(candidates_name)} artists by measuring the possibilities that I would like to listen next most, according to my preference. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. Rank directly, you can only generate artist names. You MUST rank the given candidate artists. You can not generate artists that are not in the given candidate list. You MUST not use * or () unless in candidate artist name."
        elif (self.dataset == "ml-100k" or self.dataset == "ml-1m"):
            prompt_nir = f"You are a movie recommender system on movielens dataset.\n" \
                    f"{user_his_answer}\n" \
                    f"Now there are {len(candidates_name)} candidate movies that I can watch next:\n{candidates_name}\n" \
                    f"Please rank these {len(candidates_name)} movies by measuring the possibilities that I would like to watch next most, according to my preference. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. Rank directly, you can only generate movie names. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."
        elif (self.dataset == "steam" or self.dataset == "Amazon-Games"):
            prompt_nir = f"You are a game recommender system.\n" \
                    f"{user_his_answer}\n" \
                    f"Now there are {len(candidates_name)} candidate games that I can watch next:\n{candidates_name}\n" \
                    f"Please rank these {len(candidates_name)} games by measuring the possibilities that I would like to play next most, according to my preference. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. Rank directly, you can only generate game names. You MUST rank the given candidate games. You can not generate games that are not in the given candidate list. You MUST not use * or () unless in candidate game name."
        return prompt_nir

    def predict_rank(self):
        prompt_list_his = []
        prompt_list = []
        user_his_name_list, candidates_name_list, truth_name = self.get_item_names()

        for i in range(self.batch_size):
            user_his_name = user_his_name_list[i]
            candidates_name = candidates_name_list[i]
            prompt_his = self.build_prompt_his(user_his_name, candidates_name)
            prompt_list_his.append(prompt_his)
        request_list_1 = self.request_api(prompt_list_his)

        for i in range(self.batch_size):
            user_his_name = user_his_name_list[i]
            candidates_name = candidates_name_list[i]
            prompt = self.build_prompt_rec(user_his_name, candidates_name, request_list_1[i])
            prompt_list.append(prompt)
        request_list_3 = self.request_api(prompt_list)

        results_data = []
        for i in range(self.batch_size):
            result_json = {"PID": i,
                    "ground truth": truth_name[i],
                    "Input_prefer": prompt_list_his[i],
                    "Input_rec": prompt_list[i],
                    "Predictions_prefer": request_list_1[i],
                    "Predictions_rec": request_list_3[i]}
            results_data.append(result_json)

        file_dir = f"./logs/{self.dataset}/Prefer_{self.dataset}.json"
        with open(file_dir, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        predict_text_list = []
        for request_text in request_list_3:
            predict_text = self.parse_text(request_text)
            predict_text_list.append(predict_text)

        predict_list = []
        for i in range(self.batch_size):
            predict = self.parse_predict(predict_text_list[i], truth_name[i])
            predict_list.append(predict)
        return predict_list, predict_text_list