# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:48:10 2021

@author: kesha
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from dataclasses import dataclass
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import re
import gc
import pickle
import os
from numpy.random import choice

run_type = "score"
run_test = False
wd = "C:/Users/kesha/Desktop/Projects/Next_Basket_Recommendations/Instacart/Run_Seq2Seq/"
raw_data_loc = "C:/Users/kesha/Desktop/Projects/Next_Basket_Recommendations/Instacart/"
aisles = pd.read_csv(raw_data_loc + "aisles.csv", converters={'product_id': str})
departments = pd.read_csv(raw_data_loc + "departments.csv", converters={'product_id': str})
order_products_prior = pd.read_csv(raw_data_loc + "order_products__prior.csv", converters={'product_id': str})
order_products_train = pd.read_csv(raw_data_loc + "order_products__train.csv", converters={'product_id': str})
orders = pd.read_csv(raw_data_loc + "orders.csv", converters={'user_id': str, 'order_dow': str, 'order_hour_of_day': str})
orders['days_since_prior_order'] = orders['days_since_prior_order'].fillna(999).astype(int).astype(str)
products = pd.read_csv(raw_data_loc + "products.csv", converters={'product_id': str, 'aisle_id': str, 'department_id': str})
products.drop(["product_name"], axis=1, inplace=True)

order_products_prior = pd.merge(order_products_prior, orders, how="left", on=["order_id"])
order_products_train = pd.merge(order_products_train, orders, how="left", on=["order_id"])

items_list = ['',"sot","eot", "sep", "None"] + products['product_id'].unique().tolist()
cids_list = [''] + orders['user_id'].unique().tolist()
days_list = ['', "sep"] + orders['days_since_prior_order'].unique().tolist()
aisles_list = ['', "sep"] + products['aisle_id'].unique().tolist()
department_list = ['', "sep"] + products['department_id'].unique().tolist()
dow_list = ['', "sep"] + orders['order_dow'].unique().tolist()
hod_list = ['', "sep"] + orders['order_hour_of_day'].unique().tolist()

def tokenization(list_obj):
    t = Tokenizer(oov_token='')
    word_index = dict()
    index_word = dict()
    for i, j in enumerate(list_obj):
        word_index[j] = i
        index_word[i] = j
    t.word_index = word_index
    t.index_word = index_word
    return t

if run_type == "model":
    # Items
    tokenizer_items = tokenization(items_list)
    print(tokenizer_items.word_index["sot"])
    print(tokenizer_items.word_index["eot"])
    print(len(tokenizer_items.word_index))
    
    # Transaction Rank
    tokenizer_days = tokenization(days_list)
    
    # Customer ID
    tokenizer_cids = tokenization(cids_list)
    print(len(tokenizer_cids.word_index))
    
    # Aisles
    tokenizer_aisles = tokenization(aisles_list)
    
    # Department
    tokenizer_department = tokenization(department_list)
    
    # DOW
    tokenizer_dow = tokenization(dow_list)
    
    # HOD
    tokenizer_hod = tokenization(hod_list)
    
    # Pickle the config and weights
    pickle.dump({'items': tokenizer_items,
                 'days': tokenizer_days,
                 'cids': tokenizer_cids,
                 'aisles': tokenizer_aisles,
                 'department': tokenizer_department,
                 'dow': tokenizer_dow,
                 'hod': tokenizer_hod}
                , open(wd+"tokenizer.pkl", "wb"))


def str_concat(x, add_sep = False):
    a = []
    if add_sep:
        tmp = [i for i in x if i != ''] + ["sep"]
    else:
        tmp = [i for i in x if i != '']
    a.append(tmp)
    return a


def create_universe(order_products_s, run_type = "model", orders_s = None):
    if run_type == "model":
        reorder_df = order_products_s[order_products_s.columns.tolist()].copy(deep=True)
        reorder_df['product_id'] = np.where(reorder_df['reordered']==1, reorder_df['product_id'], '')
        reorder_df = reorder_df.sort_values(['order_number','add_to_cart_order'], ascending=True).groupby(['user_id', 'order_id', 'order_number']).agg({'product_id': lambda x: str_concat(x), 'days_since_prior_order': lambda x: max(x), 'order_dow': lambda x: max(x), 'order_hour_of_day': lambda x: max(x)}).reset_index(drop=False) # .astype(int).astype(str)
        reorder_df.drop(['user_id', 'order_number'], axis=1, inplace=True)
        reorder_df.rename(columns={'product_id': 'NEXT_TXN', 'days_since_prior_order': 'NEXT_TXN_DAYS', 'order_dow': 'NEXT_TXN_DOW', 'order_hour_of_day': 'NEXT_TXN_HOD'}, inplace=True)
        reorder_df['NEXT_TXN'] = reorder_df['NEXT_TXN'].apply(lambda x : ' '.join(map(str, [i for j in x for i in j])) if len(x[0]) > 0 else 'None')
        reorder_df['NEXT_TXN'] = 'sot ' + reorder_df['NEXT_TXN'] + ' eot'
        print("Processed DV")
    
    
    order_products_s = order_products_s.sort_values(['order_number','add_to_cart_order'], ascending=True).groupby(['user_id', 'order_id', 'order_number']).agg({'product_id': lambda x: str_concat(x, add_sep = True), 'days_since_prior_order': lambda x: str_concat(x, add_sep = True), 'order_dow': lambda x: str_concat(x, add_sep = True), 'order_hour_of_day': lambda x: str_concat(x, add_sep = True), 'aisle_id': lambda x: str_concat(x, add_sep = True), 'department_id': lambda x: str_concat(x, add_sep = True)}).reset_index(drop=False) # .fillna(999).astype(int).astype(str)
    print("String Concatenation Done")
    order_products_s['TXN_HISTORY'] = order_products_s.sort_values(['order_number'], ascending=True).groupby(['user_id'])['product_id'].apply(lambda x: x.cumsum())
    order_products_s.drop(['product_id'], axis=1, inplace=True) 
    print("TXN_HISTORY Created")
    order_products_s['TXN_HISTORY_DAYS'] = order_products_s.sort_values(['order_number'], ascending=True).groupby(['user_id'])['days_since_prior_order'].apply(lambda x: x.cumsum())
    order_products_s.drop(['days_since_prior_order'], axis=1, inplace=True)
    print("TXN_HISTORY_DAYS Created")
    order_products_s['TXN_HISTORY_DOW'] = order_products_s.sort_values(['order_number'], ascending=True).groupby(['user_id'])['order_dow'].apply(lambda x: x.cumsum())
    order_products_s.drop(['order_dow'], axis=1, inplace=True)
    print("TXN_HISTORY_DOW Created")
    order_products_s['TXN_HISTORY_HOD'] = order_products_s.sort_values(['order_number'], ascending=True).groupby(['user_id'])['order_hour_of_day'].apply(lambda x: x.cumsum())
    order_products_s.drop(['order_hour_of_day'], axis=1, inplace=True)
    print("TXN_HISTORY_HOD Created")
    order_products_s['TXN_HISTORY_AISLE'] = order_products_s.sort_values(['order_number'], ascending=True).groupby(['user_id'])['aisle_id'].apply(lambda x: x.cumsum())
    order_products_s.drop(['aisle_id'], axis=1, inplace=True)
    print("TXN_HISTORY_AISLE Created")
    order_products_s['TXN_HISTORY_DEPT'] = order_products_s.sort_values(['order_number'], ascending=True).groupby(['user_id'])['department_id'].apply(lambda x: x.cumsum())
    order_products_s.drop(['department_id'], axis=1, inplace=True)
    print("TXN_HISTORY_DEPT Created")
    order_products_s['TXN_HISTORY'] = order_products_s['TXN_HISTORY'].apply(lambda x : ' '.join(map(str, [i for j in x for i in j])) if len(x)>0 else '')
    order_products_s['TXN_HISTORY_DAYS'] = order_products_s['TXN_HISTORY_DAYS'].apply(lambda x : ' '.join(map(str, [i for j in x for i in j])) if len(x[0]) > 0 else '')
    order_products_s['TXN_HISTORY_DOW'] = order_products_s['TXN_HISTORY_DOW'].apply(lambda x : ' '.join(map(str, [i for j in x for i in j])) if len(x[0]) > 0 else '')
    order_products_s['TXN_HISTORY_HOD'] = order_products_s['TXN_HISTORY_HOD'].apply(lambda x : ' '.join(map(str, [i for j in x for i in j])) if len(x[0]) > 0 else '')
    order_products_s['TXN_HISTORY_AISLE'] = order_products_s['TXN_HISTORY_AISLE'].apply(lambda x : ' '.join(map(str, [i for j in x for i in j])) if len(x[0]) > 0 else '')
    order_products_s['TXN_HISTORY_DEPT'] = order_products_s['TXN_HISTORY_DEPT'].apply(lambda x : ' '.join(map(str, [i for j in x for i in j])) if len(x[0]) > 0 else '')

    if run_type == "model":
        print("Joining Tables")
        order_products_s = pd.merge(order_products_s, reorder_df, how='left', on=['order_id']).fillna({'NEXT_TXN': 'None'})
        del reorder_df
        gc.collect()
        print("Universe Created")
        order_products_s[['NEXT_TXN', 'NEXT_TXN_DAYS', 'NEXT_TXN_DOW', 'NEXT_TXN_HOD']] = order_products_s.sort_values('order_number', ascending=True).groupby(['user_id'])['NEXT_TXN', 'NEXT_TXN_DAYS', 'NEXT_TXN_DOW', 'NEXT_TXN_HOD'].shift(-1)
        # order_products_s['NEXT_TXN'] = order_products_s.sort_values('order_number', ascending=True).groupby(['user_id'])['NEXT_TXN'].shift(-1)
        # order_products_s['NEXT_TXN_DAYS'] = order_products_s.sort_values('order_number', ascending=True).groupby(['user_id'])['NEXT_TXN_DAYS'].shift(-1)
        # order_products_s['NEXT_TXN_DOW'] = order_products_s.sort_values('order_number', ascending=True).groupby(['user_id'])['NEXT_TXN_DOW'].shift(-1)
        # order_products_s['NEXT_TXN_HOD'] = order_products_s.sort_values('order_number', ascending=True).groupby(['user_id'])['NEXT_TXN_HOD'].shift(-1)
        order_products_s = order_products_s[order_products_s['NEXT_TXN'].notna()]
    else:
        order_products_s = order_products_s.sort_values('order_number', ascending=False).drop_duplicates(['user_id'])
        orders_s.rename(columns={'days_since_prior_order': 'NEXT_TXN_DAYS', 'order_dow': 'NEXT_TXN_DOW', 'order_hour_of_day': 'NEXT_TXN_HOD'}, inplace=True)
        orders_s = orders_s.sort_values('order_number', ascending=False).drop_duplicates(['user_id'])
        order_products_s = pd.merge(order_products_s, orders_s[["user_id", "NEXT_TXN_DAYS", "NEXT_TXN_DOW", "NEXT_TXN_HOD"]], how="left", on=["user_id"])
        
    print("Done")
    return order_products_s

if run_test:
    if run_type == "model":
        # Model
        order_products_train_s = order_products_train[order_products_train['user_id']=='3'].reset_index(drop=True)
        order_products_prior_s = order_products_prior[order_products_prior['user_id']=='3'].reset_index(drop=True)
        orders_s = orders[orders['user_id']=='3'].reset_index(drop=True)
        order_products_s = pd.concat([order_products_prior_s, order_products_train_s], axis=0).reset_index(drop=True)
        order_products_s = pd.merge(order_products_s, products, how="left", on=["product_id"])
        universe_s = create_universe(order_products_s, run_type = "model", orders_s = None)
    else:   
        # Score
        order_products_train_s = order_products_train[order_products_train['user_id']=='3'].reset_index(drop=True)
        order_products_prior_s = order_products_prior[order_products_prior['user_id']=='3'].reset_index(drop=True)
        orders_s = orders[orders['user_id']=='3'].reset_index(drop=True)
        order_products_s = order_products_prior_s[~order_products_prior_s['user_id'].isin(order_products_train['user_id'])].reset_index(drop=True)
        order_products_s = pd.merge(order_products_s, products, how="left", on=["product_id"])
        universe_s = create_universe(order_products_s, run_type = "score", orders_s = orders_s)

else:
    if run_type == "model":
        # Model
        order_products = pd.concat([order_products_prior, order_products_train], axis=0).reset_index(drop=True)
        del order_products_prior, order_products_train, orders
        gc.collect()
        order_products = pd.merge(order_products, products, how="left", on=["product_id"])
        universe = create_universe(order_products, run_type = "model")
        del order_products
        gc.collect()
        universe.to_csv(wd + "model_universe.csv", index=False, chunksize=10)
    else:
        # Scoring
        order_products = order_products_prior[~order_products_prior['user_id'].isin(order_products_train['user_id'])].reset_index(drop=True)
        del order_products_prior, order_products_train
        gc.collect()
        order_products = pd.merge(order_products, products, how="left", on=["product_id"])
        universe = create_universe(order_products, run_type = "score", orders_s = orders)
        del orders, order_products
        gc.collect()
        universe.to_csv(wd + "score_universe.csv", index=False, chunksize=10)
