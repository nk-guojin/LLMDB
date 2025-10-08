import pandas as pd
from sklearn.model_selection import train_test_split
import tqdm
import random

def sort_data(file_path):
    df = pd.read_csv(f"{file_path}.inter", sep='\t', 
                    usecols=['user_id:token', 'product_id:token', 'timestamp:float']
                    )
    
    df = df.rename(columns={
        'user_id:token': 'user_id',
        'product_id:token': 'item_id',
        'timestamp:float': 'timestamp'
    })

    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 5].index

    df_filtered = df[df['user_id'].isin(valid_users)]

    df_filtered = df_filtered.astype({
        'user_id': 'int32',
        'timestamp': 'int64'
    })

    df_sorted = df_filtered.sort_values(by=['user_id', 'timestamp'])
    df_sorted.to_csv(f"{file_path}.sorted", sep=' ', index=False, header=False)
    return

def get_data(file_path):
    df = pd.read_csv(f"{file_path}.sorted", sep='\s+', usecols=[0, 1, 2], names=['user_id', 'item_id', 'timestamp'])
    
    train_data, valid_data, test_data = [], [], []
    for user_id, group in df.groupby('user_id'):
        group_sorted = group.sort_values('timestamp') 
        if len(group_sorted) < 2: continue 

        test = group_sorted.iloc[-1:]
        valid = group_sorted.iloc[-2:-1]
        train = group_sorted.iloc[:-1]
        
        train_data.append(train)
        valid_data.append(valid)
        test_data.append(test)

    train_data = pd.concat(train_data)
    valid_data = pd.concat(valid_data)
    test_data = pd.concat(test_data)
    return train_data, valid_data, test_data

def get_candidate(file_path, num_user):
    df = pd.read_csv(f"{file_path}.sorted", sep='\s+', usecols=[0, 1, 2], names=['user_id', 'item_id', 'timestamp'])
    all_users = df['user_id'].unique()
    selected_users = random.sample(list(all_users), num_user)
    
    all_items = df['item_id'].unique()
    candidates = []
    for user in selected_users:
        item_list = random.sample(list(all_items), 100)
        candidates.append([user, item_list])
    
    with open(f"{file_path}.candidate", 'w') as f:
        for user, items in candidates:
            f.write(f"{user} {' '.join(map(str, items))}\n")
    return

def generate_candidate(file_path, test_data, num_candidate):
    candidate_dict = {}
    with open(f"{file_path}.candidate", 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            user_id, items = parts[0], parts[1:]
            candidate_dict[user_id] = items
    
    df = pd.read_csv(f"{file_path}.sorted", sep=' ', usecols=[0, 1, 2], names=['user_id', 'item_id', 'timestamp'])
    user_history = df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    test_pos = test_data.set_index('user_id')['item_id'].to_dict()
    
    candidate_list = {}
    user_list = []
    for user_id in candidate_dict:
        if user_id not in test_pos or user_id not in user_history:
            continue
        user_list.append(user_id)
        interacted = user_history[user_id]
        valid_candidates = [item for item in candidate_dict[user_id] if item not in interacted]
        
        num_neg = num_candidate - 1
        if len(valid_candidates) >= num_neg:
            selected_neg = random.sample(valid_candidates, num_neg)
        else:
            selected_neg = valid_candidates.copy()
            selected_neg += random.choices(list(interacted), k=(num_neg - len(valid_candidates)))
        
        pos_item = test_pos[user_id]
        selected_neg.append(pos_item)
        
        candidate_list[user_id] = selected_neg
    
    return candidate_list, user_list

def load_text(file_path):
    item_name = {}
    with open(f"{file_path}.item", 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            v = line.strip().split('\t')
            item_name[v[4]] = v[0]

    return item_name

def inter_his(file_path, history_num=20):
    candidate_users = []
    with open(f"{file_path}.candidate", 'r') as f:
        for line in f:
            user_id = int(line.strip().split()[0])
            candidate_users.append(user_id)
    
    df = pd.read_csv(f"{file_path}.sorted", sep=' ', usecols=[0, 1, 2], names=['user_id', 'item_id', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    target_df = df[df['user_id'].isin(candidate_users)]
    
    user_groups = {user: group for user, group in target_df.groupby('user_id')}
    
    user_his_list = {}
    ground_truth = []
    for user_id in candidate_users:
        if user_id not in user_groups:
            continue 
        sorted_items = user_groups[user_id].sort_values('timestamp', ascending=True)
        recent_history = sorted_items.iloc[-1 * history_num - 1 : -1]['item_id'].tolist()
        user_his_list[user_id] = recent_history
        ground_truth.append(int(sorted_items.iloc[-1:]['item_id']))
    
    return user_his_list, ground_truth

def item_popularity(file_path, item_names):
    df = pd.read_csv(f"{file_path}.sorted", sep='\s+', usecols=[0, 1, 2], names=['user_id', 'item_id', 'timestamp'])
    item_list = list(df['item_id'])
    item_pop = {}
    for item_id in item_list:
        name = item_names[str(item_id)]
        item_pop[name] = item_pop.get(name, 0) + 1
    return item_pop