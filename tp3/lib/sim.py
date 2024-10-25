import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_labels_groups(labels, id_list=None):
    ''''
    Function to get the groups of messages based on the labels
    labels: list of labels received from the clustering algorithm AgglomerativeClustering
    [1,2,3,2,2,1] -> {1: [0, 5], 2: [1, 3, 4], 3: [2]}
    and with id_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    -> {1: [10, 60], 2: [20, 40, 50], 3: [30]}
    
    By default, the function will use the index as the id
    If id_list is provided, the function will use the id_list to get the id
    in the index position
    '''
    positions_dict = {}
    if id_list is not None:
        def get_id(index):
            return id_list[index]
    else:
        def get_id(index):
            return index

    for index, value in enumerate(labels):
        id = get_id(index)
        if value in positions_dict:
            positions_dict[value].append(id)
        else:
            positions_dict[value] = [id]
    return positions_dict


def init_similarity_dict(df_comp, ids_check=[]):
    # Get unique ids
    ids = pd.unique(df_comp[['id_1', 'id_2']].values.ravel())
    id_to_index = {id_val: i for i, id_val in enumerate(ids)}

    set_ids_check = set(ids_check)
    intersection_ids = set_ids_check.intersection(ids)
    if set_ids_check == intersection_ids:
        print("All ids in ids_check are in the df_comp")
    else:
        print(f"Ids in ids_check not in df_comp: {set_ids_check - intersection_ids}")

    # Create row, column, and distance values
    row = [id_to_index[id1] for id1 in df_comp['id_1']]
    col = [id_to_index[id2] for id2 in df_comp['id_2']]
    distances = df_comp['similarity']

    sim_dict = df_comp[['id_1', 'id_2', 'similarity']].to_dict(orient='records')
    # print(sim_dict)
    # Create a sparse matrix in CSR format
    n = len(ids)  # Total number of unique ids
    matrix = np.zeros((n, n))

    # print(f"Matrix shape: {matrix.shape}")

    # loop over sim_dict
    for sim in sim_dict:
        # print(sim)
        id1 = sim['id_1']
        id2 = sim['id_2']
        sim_val = sim['similarity']
        i = id_to_index[id1]
        j = id_to_index[id2]
        # print(f"i: {i}, j: {j}, sim: {sim_val}")
        matrix[i, j] = sim_val
        matrix[j, i] = sim_val

    return {'id_to_index': id_to_index,
            "ids": ids,
            "sim_matrix": matrix}


from sklearn.cluster import AffinityPropagation, AgglomerativeClustering

# for each group of messages, get the clusters
# 1. get the labels
# 2. get the groups
# 3. get the messages in each group
# 4. get the messages in each cluster

# loop over the groups
# for each group, get the clusters


def gen_cluster(df_comp, method='AffinityPropagation'):
    # loop over df_grouped
    results = []
    ctrl_duplicated = {}
    tokenization = {}

    # group df_comp by message_lbr, word_bin and trade_type
    df_grouped = df_comp.groupby(['message_lbr', 'word_bin', 'trade_type'], dropna=False)
    # transform the groupby object to a DataFrame
    df_grouped = df_grouped.size().reset_index(name='count')
    
    clusters = []
    # for idx, row in df_grouped2.head(1).iterrows():
    for idx, row in df_grouped.iterrows():
        # print the row
        message_lbr= row['message_lbr']
        word_bin= row['word_bin']
        trade_type= row['trade_type']
        count= row['count']
        print(f"message_lbr: {message_lbr}, word_bin: {word_bin}, trade_type: {trade_type}, count: {count}")

        # filter df by message_lbr, word_bin and commodity 
        if trade_type == '{Not Specified}':
            flt = (df_comp['message_lbr'] == message_lbr) & (df_comp['word_bin'] == word_bin) & df_comp['trade_type'].isnull()
        else:
            flt = (df_comp['message_lbr'] == message_lbr) & (df_comp['word_bin'] == word_bin) & (df_comp['trade_type'] == trade_type)
        
        grp_src = df_comp[flt]
        sim_dict = init_similarity_dict(grp_src, [11921, 10405]) # TODO - fix use: init_similarity_dict
        sim_matrix = sim_dict['sim_matrix']
        if method == 'AgglomerativeClustering':
            build_gpr = AgglomerativeClustering(metric='precomputed', linkage='complete', compute_full_tree=True, n_clusters=None, distance_threshold=0.5)
            # cur_labels = build_gpr.fit_predict(1- sim_matrix.toarray())
            cur_labels = build_gpr.fit_predict(1- sim_matrix)
        elif method == 'AffinityPropagation':
            build_gpr = AffinityPropagation(affinity='precomputed', random_state=42, damping = 0.8)
            # print(sim_matrix.toarray())
            # print(sim_matrix)
            # build_gpr.fit(sim_matrix.toarray().astype(np.float64))
            build_gpr.fit(sim_matrix.astype(np.float64))
            cur_labels = build_gpr.labels_
        else:
            raise ValueError(f"Method {method} not implemented")
            
        # print("distance_matrix")
        # print(1- sim_matrix.toarray())
        
        try:
            grps_cluster = get_labels_groups(cur_labels, sim_dict['ids'])
        except Exception as e:
            return cur_labels, sim_dict['ids'], sim_dict['sim_matrix']
        grp_ids_cluster = [v for k, v in grps_cluster.items() if len(v) > 1]
        # print(grp_ids_cluster)
        clusters.append(grp_ids_cluster)
    return clusters

# buid compare list per groups
def build_compare_list(df_grouped, window_size, df_sim):
    # loop over df_grouped
    results = []
    ctrl_duplicated = {}
    # tokenization = {}
    # msg_fld = 'message_text_clean_utf8'
    # window_size = 7

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # mdl = BertModel.from_pretrained('bert-base-uncased')

    # for idx, row in df_grouped2.head(1).iterrows():
    for idx, row in df_grouped.iterrows():
        # print the row

        message_lbr = row['message_lbr']
        word_bin = row['word_bin']
        trade_type = row['trade_type']
        count = row['count']


        print(f"message_lbr: {message_lbr}, word_bin: {word_bin}, trade_type: {trade_type}, count: {count}")

        # filter df by message_lbr, word_bin and commodity 
        if trade_type == '{Not Specified}':
            flt = (df_sim['message_lbr'] == message_lbr) & (df_sim['word_bin'] == word_bin) & df_sim['trade_type'].isnull()
        else:
            flt = (df_sim['message_lbr'] == message_lbr) & (df_sim['word_bin'] == word_bin) & (df_sim['trade_type'] == trade_type)
        
        # also filter by chat_id not in the group
        
        grp_src = df_sim[flt][['id_msg', 'chat_id', 'date_source_posted_at']].sort_values(by='date_source_posted_at')
        
        
        chk = [12058, 11994, 12068, 12070, 12072, 12077 ]
        # loop over grp_src
        for idx_src, row2 in grp_src.iterrows():
            # print the row
            id_src = row2['id_msg']
            chat_id = row2['chat_id']
            date_source_posted_at_src= row2['date_source_posted_at']
            # print(f"    id: {idx_src}, date_source_posted_at: {date_source_posted_at_src}")
            # define the slinding window
            window_start = date_source_posted_at_src
            window_end = window_start + pd.Timedelta(days=window_size)
            
            sliding_window_flt = (grp_src['date_source_posted_at'] >= window_start) &  (grp_src['date_source_posted_at'] <= window_end)
            sliding_window_flt = sliding_window_flt & (grp_src['chat_id'] != chat_id)
            
            # Select all rows in the same sub-group that are within the 7-day window
            df_sliding_window = grp_src[sliding_window_flt]
            # print the number of rows in the sliding window
            # print(f"        Number of rows in the sliding window: {df_sliding_window.shape[0]}")
            # Iterate ove the sliding window to print the id and date_source_posted_at

            for idx_comp, row3 in df_sliding_window.iterrows():
                id_comp= row3['id_msg']
                # if is the record itself, skip
                if id_src == id_comp:
                    # tokenization[id_src] =  get_bert_embedding(row3[msg_fld], tokenizer, mdl)
                    continue
                # if we already processed this comparison, skip
                if id_comp in ctrl_duplicated and id_src in ctrl_duplicated[id_comp]:
                    continue
                
                # if id_comp not in tokenization:
                #     tokenization[id_comp] =  get_bert_embedding(row3[msg_fld], tokenizer, mdl)
                # similarity = cosine(tokenization[id_src], tokenization[id_comp])
                if id_src in ctrl_duplicated:
                    ctrl_duplicated[id_src][id_comp] = 1
                else:
                    ctrl_duplicated[id_src] = {id_comp: 1}
                date_source_posted_at_comp= row3['date_source_posted_at']
                # print(f"            id_src: {id_src}, date_source_posted_at: {date_source_posted_at}")
                # print(f"            id_comp: {id_comp}, date_source_posted_at: {date_source_posted_at_comp}")
                # print("---------")
                dict_row = {
                    'message_lbr': message_lbr,
                    'word_bin': word_bin,
                    'trade_type': trade_type,
                    'id_1': id_src,
                    'id_2': id_comp,
                    'date_id_1': date_source_posted_at_src,
                    'date_id_2': date_source_posted_at_comp,
                    # 'similarity': similarity,
                    # 'distance': 1 - similarity
                    }
                if id_src in chk:
                    print('###########')
                    print(f"    lbr: {message_lbr}")
                    print(f"    word_bin: {word_bin}")
                    print(f"    trade_type: {trade_type}")
                    print('-----')
                    print(dict_row)
                    print('###########')
                    
                results.append(dict_row)

    df_results = pd.DataFrame.from_dict(results)
    return df_results


# inpunts:
# df_comp: DataFrame with the pairs of messages to compare
# tfidf_matrix: matix with the tfidf values for each message of the whole corpus - reindexed by id_msg
# id_to_index: dictionary with the index of the message in the tfidf_matrix

# output:
# - similarities2 - list with the similarities between the pairs of messages

import math
from time import time

from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Parallelized function to compute cosine similarity for a chunk of data
def compute_similarity_chunk(chunk, id_to_index, tfidf_matrix):
    similarities = []
    for _, row in chunk.iterrows():
        id1, id2 = row['id_1'], row['id_2']
        
        # Get the indices of the messages in the TF-IDF matrix
        idx1, idx2 = id_to_index[id1], id_to_index[id2]
        
        # Compute cosine similarity for the pair
        sim = cosine_similarity(tfidf_matrix[idx1], tfidf_matrix[idx2])[0][0]
        
        # Append the similarity result
        similarities.append(sim)
    return similarities

# Function to split dataframe into chunks
def split_dataframe(df, chunk_size):
    num_chunks = math.ceil(len(df) / chunk_size)
    return [df[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]


def get_similarities(df_comp, tfidf_matrix, df_sim):
    # Assume df_sim contains: id_msg, cleaned_sim (cleaned messages)
    # Assume df_results contains: id_1, id_2 (pairs of message IDs)

    # df_comp = df_results_jan_mar[['id_1', 'id_2']] 

    # Generate tfidf_matrix
    # tfidf_vectorizer = TfidfVectorizer()
    # tfidf_matrix = tfidf_vectorizer.fit_transform(df_sim['cleaned_sim'])
    id_to_index = {idx: i for i, idx in enumerate(df_sim['id_msg'])}

    # Set the chunk size and number of parallel jobs (cores)
    chunk_size = 10000  # You can adjust this depending on your system's memory
    n_jobs = -1  # Use all available cores (-1 means using all cores)

    # Split df_comp into chunks
    chunks = split_dataframe(df_comp, chunk_size)

    # Measure start time
    start = time()

    # Parallel computation
    results = Parallel(n_jobs=n_jobs)(delayed(compute_similarity_chunk)(chunk, id_to_index, tfidf_matrix) for chunk in chunks)

    # Flatten the results into a single list
    similarities = [sim for chunk_sims in results for sim in chunk_sims]

    end = time()
    elapsed = end - start

    return similarities



def enrich_chat_info(df_comp, df_sim):
    # get the messages with the ids in the df_comp using joins not changing index in the df_sim
    df_comp['id_1'] = df_comp['id_1'].astype(int)
    df_comp['id_2'] = df_comp['id_2'].astype(int)

    # drop the columns if they exist
    if 'message_text_clean_utf8_1' in df_comp.columns:
        df_comp = df_comp.drop(columns=['message_text_clean_utf8_1'])
    if 'message_text_clean_utf8_2' in df_comp.columns:
        df_comp = df_comp.drop(columns=['message_text_clean_utf8_2'])

    if 'cleaned_sim_1' in df_comp.columns:
        df_comp = df_comp.drop(columns=['cleaned_sim_1'])
    if 'cleaned_sim_2' in df_comp.columns:
        df_comp = df_comp.drop(columns=['cleaned_sim_2'])
        
    if 'date_id_1' in df_comp.columns:
        df_comp = df_comp.drop(columns=['date_id_1'])
    if 'date_id_2' in df_comp.columns:
        df_comp = df_comp.drop(columns=['date_id_2'])

    if 'chat_id_1' in df_comp.columns:
        df_comp = df_comp.drop(columns=['chat_id_1'])

    if 'chat_id_2' in df_comp.columns:
        df_comp = df_comp.drop(columns=['chat_id_2'])

    # Merge the df_comp with the df_sim to get the message text for the pairs of messages
    # if chat_id_in in df_comp.columns then drop 'message_text_clean_utf8_1', 'cleaned_sim_1', 'chat_id_1'

    df_comp = df_comp.merge(df_sim[['id_msg','message_text_clean_utf8', 'cleaned_sim', 'chat_id', 'date_source_posted_at']], left_on='id_1', right_on='id_msg',
                            how='left')

    df_comp = df_comp.rename(columns={'message_text_clean_utf8': 'message_text_clean_utf8_1', 
                                    'cleaned_sim': 'cleaned_sim_1', 'chat_id': 'chat_id_1',
                                    'date_source_posted_at': 'date_id_1'})
    df_comp = df_comp.drop(columns=['id_msg'])



    df_comp = df_comp.merge(df_sim[['id_msg','message_text_clean_utf8', 'cleaned_sim', 'chat_id', 'date_source_posted_at']], left_on='id_2', right_on='id_msg', 
                            how='left')
    df_comp = df_comp.rename(columns={'message_text_clean_utf8': 'message_text_clean_utf8_2', 
                                    'cleaned_sim': 'cleaned_sim_2', 'chat_id': 'chat_id_2',
                                    'date_source_posted_at': 'date_id_2'})
    df_comp = df_comp.drop(columns=['id_msg'])

    
    return df_comp


# Funcion para comparar los resultados de similitud para grupos de mensajes
def check_sim(grp_ids, sim_matrix_df_dict, messages_df, columns=["cleaned"]):
  for grp in grp_ids:
    print("------")
    print(f"Grupo: {grp}")
    flt = messages_df["id_msg"].isin(grp)
    tmp_df = messages_df[flt].sort_values("id_msg")
    # loop over tmp_df to print the df row
    for idx, row in tmp_df.iterrows():
      print(f" --- msg_id: {row['id_msg']} ---")
      print(f"{row['message_lbr']} - {row['word_bin']} - {row['trade_type']}")
      print(f" ----------")
      for column in columns:
        print(f"     ------")
        print(f"{row[column]}")
      # print(f"{row['message_text_clean_utf8']}")



    # for msg_id, msg in zip(tmp_df["id_msg"], tmp_df[column]):
    #   print(f" --- msg_id: {msg_id} ---")
    #   print(f"{msg}")
    #   print(f" ----------")
    df_comp = sim_matrix_df_dict['Cosine']
    print()
    combs = set([frozenset([id1, id2]) for id2 in grp for id1 in grp if id1 != id2])
    # get scores:
    for pair in combs:
      ids = list(pair)
      print(f"  Pair:   {ids}")
      for name, matrix in sim_matrix_df_dict.items():
        flt = (df_comp['id_1'] == ids[0]) & (df_comp['id_2'] == ids[1]) | (df_comp['id_1'] == ids[1]) & (df_comp['id_2'] == ids[0])
        distance_val = matrix[flt]['distance'].tolist()
        # get the value of matrix[flt]['distance'] to get the distance
        print(distance_val)
        if distance_val != []:
          print(f"      {name} - {distance_val[0]}")
        else:
          print(f"      {name} - No distance calculated")
        

    print()


def gen_chat_pair_metrics(df_comp_chat_metrics):
    """
    Generate the chat pair metrics
        group idx_chats
        Required columns: idx_chats, id_1, id_2, similarity, distance, date_diff_hours
        Output columns: idx_chats, num_messages_pairs, msgs_chat_id_1, msgs_chat_id_2, 
                        sum_similarity, avg_similarity, sum_distance, avg_distance, 
                        msg_ratio, sim_ratio, sim_ratio_norm
        The function will sort the results by sim_ratio_norm
        sim_ratio_norm goes from 0 to 1, where 1 is the maximum value of sim_ratio 
        and indicates the highest similarity between the chats
    """
    # group by chat_id_1, chat_id_2, date_id_1_round and get the count of messages, the sum of similarity and the mean of distance
    df_comp_grouped = df_comp_chat_metrics.groupby(['idx_chats']).agg({
        'id_1': ['count', 'nunique'],
        'id_2': ['nunique'],
        'similarity': ['sum', 'mean'],
        'distance': ['sum', 'mean']
        # 'date_diff_hours': ['sum', 'mean']
    }).reset_index()

    # rename the columns df_comp_grouped Count -> num_messages, similarity -> sum_similarity, distance -> avg_distance
    df_comp_grouped.columns = ['idx_chats', 
                            'num_messages_pairs', 'msgs_chat_id_1', 'msgs_chat_id_2',
                            'sum_similarity', 'avg_similarity',
                            'sum_distance', 'avg_distance'
                            # 'sum_date_diff_hours', 'avg_date_diff_hours',
                            ]
    
    df_comp_grouped['msg_ratio'] = df_comp_grouped.apply(lambda x: 
                                         max(x['msgs_chat_id_1'], x['msgs_chat_id_2']) / min(x['msgs_chat_id_1'],
                                                                                             x['msgs_chat_id_2']), axis=1)

    df_comp_grouped['sim_ratio'] = df_comp_grouped['sum_similarity']/df_comp_grouped['msg_ratio']

    # nomalizar sim_ratio
    # get the max of sim_ratio
    max_sim_ratio = df_comp_grouped['sim_ratio'].max()
    df_comp_grouped['sim_ratio_norm'] = df_comp_grouped['sim_ratio'] / max_sim_ratio

    df_comp_grouped[['chat_id_low', 'chat_id_high']] = df_comp_grouped['idx_chats'].str.split('-', expand=True).astype(int)

    # sort by sim_ratio
    df_comp_grouped = df_comp_grouped.sort_values(by='sim_ratio_norm', ascending=False)

    return df_comp_grouped
