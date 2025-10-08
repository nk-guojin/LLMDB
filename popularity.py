import math

def get_pop(text, item_pop):
    pop = []
    for item_text in text:
        if item_text in item_pop:
            pop.append(item_pop[item_text])
        else:
            pop.append(0)
    return pop

def get_mean(pop_list):
    pop = []
    for j in range(20):
        num = 0
        sums = 0
        for i in pop_list:
            if len(i) > j and i[j] != 0:
                sums = sums + i[j]
                num = num + 1
            elif len(i) > j:
                continue
            else:
                continue
        if num != 0:
            pop.append(sums / num)
        else:
            pop.append(0)
    return pop

def get_user_pop(user_his, item_names, users, item_pop):
    user_his_name_list = []

    for user_id in users:
        user_his_name = []

        for his in user_his[user_id]:
            user_his_name.append(item_names[str(his)])

        user_his_name_list.append(user_his_name)
    
    pop_list = []
    for text in user_his_name_list:
        pop_list.append(get_pop(text, item_pop))
    pop = get_mean(pop_list)
    return pop

def get_log_user(user_list):
    pop_sum = 0
    for i in range(20):
        pop_sum += math.log2(user_list[i])
    pop_sum /= 20
    return round(pop_sum, 5)

def get_log_pop(pop_list, user_list):
    topK = [1, 5, 10, 20]
    topK_list = []
    total_pop = 0
    for i in range(20):
        total_pop += pop_list[i]
    for k in topK:
        pop_sum = 0
        for i in range(k):
            pop_sum += math.log2(pop_list[i])
        pop_sum /= k
        topK_list.append(pop_sum - get_log_user(user_list))
    return [round(i, 5) for i in topK_list]

def popularity(predict_text_list, item_pop):
    pop_list = []
    for text in predict_text_list:
        pop_list.append(get_pop(text, item_pop))
    pop = get_mean(pop_list)
    return pop


def get_truth_pop(gt_list, item_pop):
    gt_pop = 0
    for gt_text in gt_list:
        if gt_text in item_pop:
            gt_pop += item_pop[gt_text]
            continue
        else:
            gt_pop += 0
    gt_pop = gt_pop / len(gt_list)  
    return gt_pop
