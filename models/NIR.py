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

class NirModel(BaseModel):
    def __init__(self, user_his, candidates, item_names, users, ground_truth, dataset):
        super().__init__(user_his, candidates, item_names, users, ground_truth, dataset)

    # 用户历史趋势
    def build_prompt_his(self, user_his_name, candidates_name):
        if (self.dataset == "lastfm"):
            prompt_nir = f"Candidate Set (candidate artists):\n{candidates_name}\n\n" \
                    f"The artists I have listened (listened artists):\n{user_his_name}\n\n" \
                    f"Step 1: What features are most important to me when selecting artists (Summarize my preferences briefly)? \n" \
                    f"Answer: \n"
        elif (self.dataset == "ml-100k" or self.dataset == "ml-1m"):
            prompt_nir = f"Candidate Set (candidate movies):\n{candidates_name}\n\n" \
                    f"The movies I have watched (watched movies):\n{user_his_name}\n\n" \
                    f"Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)? \n" \
                    f"Answer: \n"
        elif (self.dataset == "steam" or self.dataset == "Amazon-Games"):
            prompt_nir = f"Candidate Set (candidate games):\n{candidates_name}\n\n" \
                    f"The games I have played (played games):\n{user_his_name}\n\n" \
                    f"Step 1: What features are most important to me when selecting games (Summarize my preferences briefly)? \n" \
                    f"Answer: \n"
        return prompt_nir

    # 用户对流行度敏感性
    def build_prompt_pre(self, user_his_name, candidates_name, user_his_answer):
        if (self.dataset == "lastfm"):
            prompt_nir = f"Candidate Set (candidate artists):\n{candidates_name}\n\n" \
                    f"The artists I have listened (listened artists):\n{user_his_name}\n\n" \
                    f"Step 1: What features are most important to me when selecting artists (Summarize my preferences briefly)? \n" \
                    f"Answer: {user_his_answer}\n" \
                    f"Step 2: Selecting the most featured artists (at most 5 artists) from the artists according to my preferences in descending order (Format: no. an artist.). \n" \
                    f"Answer: \n"
        elif (self.dataset == "ml-100k" or self.dataset == "ml-1m"):
            prompt_nir = f"Candidate Set (candidate movies):\n{candidates_name}\n\n" \
                    f"The movies I have watched (watched movies):\n{user_his_name}\n\n" \
                    f"Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)?\n" \
                    f"Answer: {user_his_answer}\n" \
                    f"Step 2: Selecting the most featured movies (at most 5 movies) from the watched movies according to my preferences in descending order (Format: no. a watched movie.). \n" \
                    f"Answer: \n"
        elif (self.dataset == "steam" or self.dataset == "Amazon-Games"):
            prompt_nir = f"Candidate Set (candidate games):\n{candidates_name}\n\n" \
                    f"The games I have played (played games):\n{user_his_name}\n\n" \
                    f"Step 1: What features are most important to me when selecting games (Summarize my preferences briefly)? \n" \
                    f"Answer: {user_his_answer}\n" \
                    f"Step 2: Selecting the most featured games (at most 5 games) from the played games according to my preferences in descending order (Format: no. a played game.). \n" \
                    f"Answer: \n"
        return prompt_nir

    # 构建提示词，通过用户历史记录对候选电影进行排名，一个用户一条prompt
    def build_prompt_rec(self, user_his_name, candidates_name, user_his_answer, user_pop_answer):
        if (self.dataset == "lastfm"):
            prompt_nir = f"Candidate Set (candidate artists):\n{candidates_name}\n\n" \
                    f"The artists I have listened (listened artists):\n{user_his_name}\n\n" \
                    f"Step 1: What features are most important to me when selecting artists (Summarize my preferences briefly)? \n" \
                    f"Answer: {user_his_answer}\n" \
                    f"Step 2: Selecting the most featured artists (at most 5 artists) from the artists according to my preferences in descending order (Format: no. an artist.). \n" \
                    f"Answer: {user_pop_answer}\n" \
                    f"Step 3: Can you recommend {len(candidates_name)} artists from the Candidate Set similar to the selected artists I've watched (Format: no. an candidate artist).\n" \
                    f"Answer: \n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. Rank directly, you can only generate artist names. You MUST rank the given candidate artists. You can not generate artists that are not in the given candidate list. You MUST not use * or () unless in candidate artist name."
        elif (self.dataset == "ml-100k" or self.dataset == "ml-1m"):
            prompt_nir = f"Candidate Set (candidate movies):\n{candidates_name}\n\n" \
                    f"The movies I have watched (watched movies):\n{user_his_name}\n\n" \
                    f"Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)?\n" \
                    f"Answer: {user_his_answer}\n" \
                    f"Step 2: Selecting the most featured movies (at most 5 movies) from the watched movies according to my preferences in descending order (Format: no. a watched movie.). \n" \
                    f"Answer: {user_pop_answer}\n" \
                    f"Step 3: Can you recommend {len(candidates_name)} movies from the Candidate Set similar to the selected movies I've watched (Format: no. a candidate movie).\n" \
                    f"Answer: \n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. Rank directly, you can only generate movie names. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list. You MUST not use * or () unless in candidate movie name."
        elif (self.dataset == "steam" or self.dataset == "Amazon-Games"):
            prompt_nir = f"Candidate Set (candidate games):\n{candidates_name}\n\n" \
                    f"The games I have played (played games):\n{user_his_name}\n\n" \
                    f"Step 1: What features are most important to me when selecting games (Summarize my preferences briefly)? \n" \
                    f"Answer: {user_his_answer}\n" \
                    f"Step 2: Selecting the most featured games (at most 5 games) from the played games according to my preferences in descending order (Format: no. a played game.). \n" \
                    f"Answer: {user_pop_answer}\n" \
                    f"Step 3: Can you recommend {len(candidates_name)} games from the Candidate Set similar to the selected games I've played (Format: no. a candidate game).\n" \
                    f"Answer: \n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. Rank directly, you can only generate game names. You MUST rank the given candidate movies. You can not generate games that are not in the given candidate list. You MUST not use * or () unless in candidate game name."
        return prompt_nir

    # 通过整合上述函数输出预测的电影序列
    def predict_rank(self):
        prompt_list_his = []
        prompt_list_pre = []
        prompt_list = []
        user_his_name_list, candidates_name_list, truth_name = self.get_item_names()

        # prompt1
        for i in range(self.batch_size):
            user_his_name = user_his_name_list[i]
            candidates_name = candidates_name_list[i]
            prompt_his = self.build_prompt_his(user_his_name, candidates_name)
            prompt_list_his.append(prompt_his)
        request_list_1 = self.request_api(prompt_list_his)

        for i in range(self.batch_size):
            user_his_name = user_his_name_list[i]
            candidates_name = candidates_name_list[i]
            prompt_pre = self.build_prompt_pre(user_his_name, candidates_name, request_list_1[i])
            prompt_list_pre.append(prompt_pre)
        request_list_2 = self.request_api(prompt_list_pre)

        # prompt3
        for i in range(self.batch_size):
            user_his_name = user_his_name_list[i]
            candidates_name = candidates_name_list[i]
            prompt = self.build_prompt_rec(user_his_name, candidates_name, request_list_1[i], request_list_2[i])
            prompt_list.append(prompt)
        request_list_3 = self.request_api(prompt_list)

        results_data = []
        for i in range(self.batch_size):
            result_json = {"PID": i,
                    "ground truth": truth_name[i],
                    "Input_his": prompt_list_his[i],
                    "Input_pre": prompt_list_pre[i],
                    "Input_rec": prompt_list[i],
                    "Predictions_his": request_list_1[i],
                    "Predictions_pre": request_list_2[i],
                    "Predictions_rec": request_list_3[i]}
            results_data.append(result_json)

        file_dir = f"./logs/{self.dataset}/NIR_{self.dataset}.json"
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