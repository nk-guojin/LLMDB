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
from utils import get_api_responses

class IterationModel(BaseModel):
    def __init__(self, user_his, candidates, item_names, users, ground_truth, dataset):
        super().__init__(user_his, candidates, item_names, users, ground_truth, dataset)

    def request_api(self, prompt_list, iter_list, last_reply):
        request_list = []
        for i in tqdm(range(len(prompt_list))):
            if iter_list[i] == -1:
                request_list.append(get_api_responses(prompt_list[i]))
                time.sleep(0.5)
            else:
                request_list.append(last_reply[i])
        return request_list
    
    def get_rbo_score(self, list1, list2, p=0.8):
        if not list1 or not list2:
            return 0
        
        s, t = dict(), dict()
        for i, e in enumerate(list1):
            s[e] = i
        for i, e in enumerate(list2):
            t[e] = i
        
        common_elements = set(s.keys()) & set(t.keys())
        
        if not common_elements:
            return 0
        
        k = max(len(list1), len(list2))
        x_d = {} 
        overlap = 0 
        depth = 0   
        
        for depth in range(1, k + 1):
            if depth <= len(list1):
                e1 = list1[depth - 1]
                if e1 in t and t[e1] < depth:
                    overlap += 1
            
            if depth <= len(list2):
                e2 = list2[depth - 1]
                if e2 in s and s[e2] < depth:
                    overlap += 1
                    if depth <= len(list1) and list1[depth - 1] == e2:
                        overlap -= 1
            
            x_d[depth] = overlap / depth
        
        rbo_score = 0
        for d in range(1, k + 1):
            rbo_score += x_d[d] * (p ** (d - 1))
        
        rbo_score *= (1 - p)
        
        return rbo_score

    def build_prompt_act(self, user_his_name, candidates_name):
        if (self.dataset == "lastfm"):

            prompt_act = f"You are a music recommender system.\n" \
                    f"I've listened musics from the following artists in the past in order:\n{user_his_name}\n\n" \
                    f"Note that my most recently listened artist is {user_his_name[-1]}.\n" \
                    f"Now there are {len(candidates_name)} candidate artists that I can listen to next:\n{candidates_name}\n" \
                    f"Please rank these {len(candidates_name)} artists by measuring the possibilities that I would like to listen next most, according to my listen history. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. Rank directly, you can only generate artist names. You MUST rank the given candidate artists. You can not generate artists that are not in the given candidate list."
        elif (self.dataset == "ml-100k" or self.dataset == "ml-1m"):
            prompt_act = f"You are a movie recommender system on movielens dataset.\n" \
                    f"I've watched the following movies in the past in order:\n{user_his_name}\n\n" \
                    f"Note that my most recently watched movie is {user_his_name[-1]}.\n" \
                    f"Now there are {len(candidates_name)} candidate movies that I can watch next:\n{candidates_name}\n" \
                    f"Please rank these {len(candidates_name)} movies by measuring the possibilities that I would like to watch next most, according to my watching history. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. Rank directly, you can only generate movie names. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."
        elif (self.dataset == "steam" or self.dataset == "Amazon-Games"):
            prompt_act = f"You are a game recommender system.\n" \
                    f"I've played games in the past in order:\n{user_his_name}\n\n" \
                    f"Note that my most recently played game is {user_his_name[-1]}.\n" \
                    f"Now there are {len(candidates_name)} candidate games that I can play next:\n{candidates_name}\n" \
                    f"Please rank these {len(candidates_name)} games by measuring the possibilities that I would like to play next most, according to my play history. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. Rank directly, you can only generate game names. You MUST rank the given candidate games. You can not generate games that are not in the given candidate list."
        return prompt_act

    def build_prompt_cri(self, user_his_name, candidates_name, act_answer):
        if (self.dataset == "lastfm"):
            prompt_cri = f"You are a music recommender system.\n" \
                    f"I've listened musics from the following artists in the past in order:\n{user_his_name}\n\n" \
                    f"Note that my most recently listened artist is {user_his_name[-1]}.\n" \
                    f"Now there are {len(candidates_name)} candidate artists that I can lsiten to next:\n{candidates_name}\n" \
                    f"A music recommender system has generated such recommendatation by measuring the possibilities that I would like to listen next most:\n" \
                    f"Answer: {act_answer}\n" \
                    f"Analyze the recommendation results in the previous step, point out its shortcomings and how to optimize them to improve recommandation quality.\n"

        elif (self.dataset == "ml-100k" or self.dataset == "ml-1m"):
            prompt_cri = f"You are a recommender system evaluate expert.\n" \
                    f"I've watched the following movies in the past in order:\n{user_his_name}\n\n" \
                    f"Note that my most recently watched movie is {user_his_name[-1]}.\n" \
                    f"Now there are {len(candidates_name)} candidate movies that I can watch to next:\n{candidates_name}\n" \
                    f"A movie recommender system has generated such recommendatation by measuring the possibilities that I would like to watch next most:\n" \
                    f"{act_answer}\n" \
                    f"Analyze the recommendation results in the previous step, point out its shortcomings and how to optimize them to improve recommendation quality.\n"
                    
        elif (self.dataset == "steam" or self.dataset == "Amazon-Games"):
            prompt_cri = f"You are a game recommender system.\n" \
                    f"I've played the following games in the past in order:\n{user_his_name}\n\n" \
                    f"Note that my most recently played game is {user_his_name[-1]}.\n" \
                    f"Now there are {len(candidates_name)} candidate games that I can play next:\n{candidates_name}\n" \
                    f"A game recommender system has generated such recommendatation by measuring the possibilities that I would like to play next most:\n" \
                    f"Answer: {act_answer}\n" \
                    f"Analyze the recommendation results in the previous step, point out its shortcomings and how to optimize them to improve recommandation quality.\n"
        return prompt_cri

    def build_prompt_rec(self, user_his_name, candidates_name, act_answer, cri_answer):
        if (self.dataset == "lastfm"):
            prompt_rec = f"You are a music recommender system.\n" \
                    f"I've listened musics from the following artists in the past in order:\n{user_his_name}\n\n" \
                    f"Note that my most recently listened artist is {user_his_name[-1]}.\n" \
                    f"Now there are {len(candidates_name)} candidate artists that I can lsiten to next:\n{candidates_name}\n" \
                    f"A music recommender system has generated such recommendatation by measuring the possibilities that I would like to listen next most:\n" \
                    f"Answer: {act_answer}\n" \
                    f"Given the feedback of recommendation: {cri_answer}.\n" \
                    f"Based on former recommendations and feedback, adjust the ranking of these {len(candidates_name)} artists by measuring the possibilities that I would like to listen next most, according to my listen history. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. Rank directly, you can only generate artist names. You MUST rank the given candidate artist. You can not generate artists that are not in the given candidate list."

        elif (self.dataset == "ml-100k" or self.dataset == "ml-1m"):
            prompt_rec = f"You are a movie recommender system on movielens dataset.\n" \
                    f"I've watched the following movies in the past in order:\n{user_his_name}\n\n" \
                    f"Note that my most recently watched movie is {user_his_name[-1]}.\n" \
                    f"Now there are {len(candidates_name)} candidate movies that I can watch next:\n{candidates_name}\n" \
                    f"A movie recommender system has generated such recommendatation by measuring the possibilities that I would like to watch next most:\n" \
                    f"{act_answer}\n" \
                    f"Given the feedback of recommendation: {cri_answer}.\n" \
                    f"Based on former recommendations and feedback, adjust the ranking of these {len(candidates_name)} movies by measuring the possibilities that I would like to watch next most, according to watching history. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. Rank directly, you can only generate movie names. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."
            
        elif (self.dataset == "steam" or self.dataset == "Amazon-Games"):
            prompt_rec = f"You are a game recommender system.\n" \
                    f"I've played the following games in the past in order:\n{user_his_name}\n\n" \
                    f"Note that my most recently played game is {user_his_name[-1]}.\n" \
                    f"Now there are {len(candidates_name)} candidate games that I can play next:\n{candidates_name}\n" \
                    f"A game recommender system has generated such recommendatation by measuring the possibilities that I would like to play next most:\n" \
                    f"{act_answer}\n" \
                    f"Given the feedback of recommendation: {cri_answer}.\n" \
                    f"Based on former recommendations and feedback, adjust the ranking of these {len(candidates_name)} games by measuring the possibilities that I would like to play next most, according to my play history. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. Rank directly, you can only generate game names. You MUST rank the given candidate games. You can not generate games that are not in the given candidate list."
        return prompt_rec

    def predict_rank(self):
        prompt_list_act_list = []
        prompt_list_cri_list = []
        prompt_list_list = []

        req_list = []
        cri_list = []

        iter_list = [-1 for i in range(self.batch_size)]

        user_his_name_list, candidates_name_list, truth_name = self.get_item_names()

        last_reply_act = []
        last_reply_cri = []
        prompt_list_act = []
        for i in range(self.batch_size):
            user_his_name = user_his_name_list[i]
            candidates_name = candidates_name_list[i]
            prompt_act = self.build_prompt_act(user_his_name, candidates_name)
            prompt_list_act.append(prompt_act)
        prompt_list_act_list.append(prompt_list_act)
        rec_list_1 = self.request_api(prompt_list_act_list[0], iter_list, last_reply_act)
        last_reply_act = rec_list_1

        req_list.append(rec_list_1)

        it_count = 0
        for it in range(0,3):
            prompt_list_cri = []
            for i in range(self.batch_size):
                user_his_name = user_his_name_list[i]
                candidates_name = candidates_name_list[i]
                prompt_cri = self.build_prompt_cri(user_his_name, candidates_name, req_list[it][i])
                prompt_list_cri.append(prompt_cri) 
            prompt_list_cri_list.append(prompt_list_cri)
            cri = self.request_api(prompt_list_cri_list[it], iter_list, last_reply_cri)
            cri_list.append(cri)
            last_reply_cri = cri

            prompt_list_act_2 = []
            for i in range(self.batch_size):
                user_his_name = user_his_name_list[i]
                candidates_name = candidates_name_list[i]
                prompt_act2 = self.build_prompt_rec(user_his_name, candidates_name, req_list[it][i], cri_list[it][i])
                prompt_list_act_2.append(prompt_act2)
            prompt_list_act_list.append(prompt_list_act_2)
            rec_list_2 = self.request_api(prompt_list_act_list[it + 1], iter_list, last_reply_act)
            req_list.append(rec_list_2)

            last_reply_act = rec_list_2

            predict_it_list = []
            predict_rec_list = []
            for request_text in req_list[it]:
                predict_text = self.parse_text(request_text)
                predict_it_list.append(predict_text)
            for request_text in req_list[it + 1]:
                predict_text = self.parse_text(request_text)
                predict_rec_list.append(predict_text)

            predict_list = []

            pre_count = 0
            rbo_list = [0 for i in range(self.batch_size)]
            for i in range(self.batch_size):
                rbo_list[i] = self.get_rbo_score(predict_it_list[i], predict_rec_list[i])
            for i in range(self.batch_size):
                if iter_list[i] == -1 and rbo_list[i] > 0.7:
                    pre_count = pre_count + 1
                    iter_list[i] = it_count
            it_count = it_count + 1

        for i in range(self.batch_size):
            if iter_list[i] == -1:
                iter_list[i] = it_count

        predict_text_list = []
        for request_text in req_list[it_count]:
            predict_text = self.parse_text(request_text)
            predict_text_list.append(predict_text)

        predict_list = []
        
        for i in range(self.batch_size):
            predict = self.parse_predict(predict_text_list[i], truth_name[i])
            predict_list.append(predict)

        predict_num_list = [-1 for i in range(self.batch_size)]
        for i in range(self.batch_size):
            for j in range(len(predict_list[i])):
                if predict_list[i][j]:
                    predict_num_list[i] = j
                    break

        results_data = []
        for i in range(self.batch_size):
            result_json = {"PID": i,
                    "rounds" : iter_list[i],
                    "ground truth": truth_name[i]}
            for it in range(iter_list[i]):
                result_json[f"Input_act_{it}"] = prompt_list_act_list[it][i]
                result_json[f"Input_cri_{it}"] = prompt_list_cri_list[it][i]
            result_json["Input_rec"] = prompt_list_act_list[iter_list[i]][i]

            for it in range(iter_list[i]):
                result_json[f"Predict_act_{it}"] = req_list[it][i]
                result_json[f"Predict_cri_{it}"] = cri_list[it][i]
            result_json["Predict_rec"] = req_list[iter_list[i]][i]
            result_json["hit_num"] = predict_num_list[i] + 1
            results_data.append(result_json)

        file_dir = f"./logs/{self.dataset}/Iteration_{self.dataset}.json"
        with open(file_dir, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        return predict_list, predict_text_list