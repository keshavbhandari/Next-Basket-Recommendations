# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:14:36 2020

@author: kbhandari
"""

# Configs
downsample_pct = 0.4
cat_max_seq_length = 120
cont_max_seq_length = 100
categorical_cols = ['StockCode', 'txn_rank_1', 'txn_rank_2', 'txn_rank_3']
autoencoder_batch_size = 256
autoencoder_epochs = 10
latent_features = 64
recommendation_batch_size = 1024
recommendation_epochs = 1
workers = 1
wd = "C:/Users/kbhandari/OneDrive - Publicis Groupe/Desktop/Senior ML Test/Market Basket Recommendations/"
fileName = "Data.csv"
runType = "model" # model or scoring
runTest = True